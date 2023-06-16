import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size
import cv2

from tool import utils 


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
        
    return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(args,imh, imw, model, img, conf_thresh, nms_thresh, use_cuda=1):

    # model.eval()  #already done

    t0 = time.time()

    # christian added but already done before. And its too slow. preprocess is done on GPU
    #img = [cv2.resize(im, (model.width, model.height)) for im in img]
    #img = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in img]
    #img = [torch.from_numpy(im.transpose(2, 0, 1)).float().div(255.0) for im in img]

    # # christian this below is already done
    # if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
    #     img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    # elif type(img) == np.ndarray and len(img.shape) == 4:
    #     img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
    # else:
    #     print("unknow image type")
    #     exit(-1)
    #img = torch.stack([im for im in img], 0)
    #if use_cuda:
    #    img = img.cuda()
    #img = torch.autograd.Variable(img)


    t1 = time.time()
    output = model(img)
    t2 = time.time()

    #print('-----------------------------------')
    #print('           Preprocess : %f' % (t1 - t0))
    #print('      Model Inference : %f' % (t2 - t1))
    #print('-----------------------------------')

    # christian : quadrant

    # segment output list into sublist of lenght args.imsplits**2 : each quadrant has args.imsplits**2 images
    output[0] = [output[0][i:i + args.imsplit ** 2] for i in range(0, len(output[0]), args.imsplit ** 2)]  #boxes
    output[1] = [output[1][i:i + args.imsplit ** 2] for i in range(0, len(output[1]), args.imsplit ** 2)]  #cat

    # shift and scale boxes for all subimages in each quadrant : no shift for subimage 0
    for k in range(len(output[0])):
        boxes = output[0][k][0]
        cat = output[1][k][0]
        for i in range(args.imsplit ** 2):
            if len(output[0][k][i]) != 0:
                output[0][k][i][:,0, 0] *= imw
                output[0][k][i][:,0, 2] *= imw
                output[0][k][i][:,0, 1] *= imh
                output[0][k][i][:,0, 3] *= imh

                m, n = np.unravel_index(i, (args.imsplit, args.imsplit))
                output[0][k][i][:,0, 0] += max(0,n * imw - 2*args.overlap) # n * imw
                output[0][k][i][:,0, 2] += max(0,n * imw - 2*args.overlap) #n * imw
                output[0][k][i][:,0, 1] += max(0,m * imh - 2*args.overlap) #m * imh
                output[0][k][i][:,0, 3] += max(0,m * imh - 2*args.overlap) #m * imh
                if i != 0 and len(output[0][k][0]) != 0:
                    boxes = torch.cat((boxes, output[0][k][i]), dim=0)
                    cat = torch.cat((cat, output[1][k][i]), dim=0)

        if k == 0 and len(boxes) != 0:
            finaboxes = boxes.unsqueeze(0)
            finalcat = cat.unsqueeze(0)
        else :
            finaboxes = torch.cat((finaboxes, boxes.unsqueeze(0)), dim=0)
            finalcat = torch.cat((finalcat, cat.unsqueeze(0)), dim=0)

    output[0] = finaboxes
    output[1] = finalcat


    return utils.post_processing(img, conf_thresh, nms_thresh, output)

