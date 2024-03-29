# Author: Zylo117

import os

import cv2
import numpy as np
import torch
from glob import glob
from torch import nn
from torchvision.ops import nms
from torchvision.ops.boxes import batched_nms
from typing import Union
import uuid
import time
from utils.sync_batchnorm import SynchronizedBatchNorm2d
from torchvision.transforms import transforms

from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_
import math
import webcolors
#from config import *
from multiprocessing import pool
from multiprocessing.dummy import Pool as ThreadPool
#args = parse_args()

def invert_affine(metas: Union[float, list, tuple], preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) == 0:
            continue
        else:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                new_w, new_h, old_w, old_h, padding_w, padding_h = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (new_w / old_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (new_h / old_h)
    return preds


def aspectaware_resize_padding(image, width, height, interpolation=None, means=None):
    old_h, old_w, c = image.shape
    if old_w > old_h:
        new_w = width
        new_h = int(width / old_w * old_h)
    else:
        new_w = int(height / old_h * old_w)
        new_h = height

    canvas = np.zeros((height, height, c), np.float32)
    if means is not None:
        canvas[...] = means

    if new_w != old_w or new_h != old_h:
        if interpolation is None:
            image = cv2.resize(image, (new_w, new_h))
        else:
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    padding_h = height - new_h
    padding_w = width - new_w


    if c > 1:
        canvas[:new_h, :new_w] = image

    else:
        if len(image.shape) == 2:
            canvas[:new_h, :new_w, 0] = image
        else:
            canvas[:new_h, :new_w] = image

    return canvas, new_w, new_h, old_w, old_h, padding_w, padding_h


def preprocess(*image_path, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    normalized_imgs = [(img / 255 - mean) / std for img in ori_imgs]
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def preprocess_video(frame_from_video, max_size=512, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
    ori_imgs = frame_from_video


    mean=np.array([0.406, 0.456, 0.485],dtype=np.float32)
    std=np.array([0.225, 0.224, 0.229],dtype=np.float32)
    normalized_imgs = [(img.astype(np.float32) / 255 - mean) / std for img in ori_imgs]
    
    imgs_meta = [aspectaware_resize_padding(img[..., ::-1], max_size, max_size,
                                            means=None) for img in normalized_imgs]
    

    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]


    return ori_imgs, framed_imgs, framed_metas


def preprocess_video_normalize(images,imgscale,imgmean,imgstd):
    mean=np.array([0.485, 0.456, 0.406],dtype=np.float32)
    std=np.array([0.229, 0.224, 0.225],dtype=np.float32)

#    normalized_imgs = list((np.asarray(images).astype(np.float32)/255-mean)/std)

#    normalized_imgs = [(img.astype(np.float32) / imgscale - imgmean) / imgstd for img in images]

    normalized_imgs1 = [(img.astype(np.float32) /np.float32(255) -mean)/std  for img in images]

#    normalized_imgs1 = [np.divide(np.subtract(np.divide(img.astype(np.float32),255), mean),std)  for img in images]

#    normalized_imgs1 = [np.zeros_like(images[0])] * len(images)
#    for i in range(len(images)):
#        normalized_imgs1[i]=(images[0].astype(np.float32) /255 -mean)/std
    
    
#    normalized_imgs2 = [(img.astype(np.float32) /255 -mean)/std  for img in images[8:]]
    
#    normalized_imgs = [(img.astype(np.float32) / 255 - mean) / std for img in images]

    return normalized_imgs1 #+ normalized_imgs2



def preprocess_video_resize(images,inputsize):
    w = images[0].shape[1]
    h = images[0].shape[0]
    
    new_h=int(h * inputsize/w)
    new_w=int(w * inputsize/w)

    resized_imgs = [cv2.resize(img[..., ::-1],(new_w,new_h)) for img in images]
    return resized_imgs


def preprocess_video_crop(images,inputsize):
    w = images[0].shape[1]
    h = images[0].shape[0]
    
    wstart = int((w-inputsize) /2)
    wend = wstart + inputsize + 0
    croped_imgs1 = [img[:,:, ::-1][: , wstart : wend , :] for img in images[0:8]]
    croped_imgs2 = [img[:,:, ::-1][: , wstart : wend , :] for img in images[8:]]
    
    
    return croped_imgs1 + croped_imgs2 




def preprocess_video_pad(images):
    canvaslefts=[]
    for i in range(len(images)):
        shape=images[i].shape
#        print(shape)
        canvasleft = np.zeros((np.max(shape), np.max(shape), 3), np.float32)
        canvasleft[:shape[0], :shape[1]] = images[i]  
        canvaslefts.append(canvasleft)
    return canvaslefts





class SquarePad:
    def __call__(self,inputsize , image):
        c, h, w = image.size()
        padding = (0, 0, 0, int(inputsize - inputsize / w * h))
        return F.pad(image, padding, 'constant', 0)


def transform_compose(image, inputsize):
    transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((int(inputsize / image.shape[1] * image.shape[0]), inputsize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                SquarePad(inputsize)
                ])

    def transform(self,images):
        transformed_imgs = [self.transform_compose(img) for img in images]
        return transformed_imgs














#def preprocessvideo(image):
#    h=int(480 * 512/800)
#    w=int(800*512/800)
#    mean=np.array([0.406, 0.456, 0.485],dtype=np.float32)
#    std=np.array([0.225, 0.224, 0.229],dtype=np.float32)
#    normalized_img = (image.astype(np.float32) / 255 - mean) / std 
#    resized_img = cv2.resize(normalized_img[..., ::-1],(w,h))
#    canvasleft = np.zeros((w, w, 3), np.float32)
#    canvasleft[:h, :w] = resized_img
#
#    return canvasleft

#def preprocess_video(images):
#
#    pool = ThreadPool(16)
##    canvaslefts=pool.apply(preprocessvideo, images)
#    canvaslefts=pool.map( preprocessvideo, images )
#    return canvaslefts

def quandrant_box_adjust(model,args,imw,imh,transformed_anchors, i):
    # shift and scale boxes for all subimages in each quadrant : no shift for subimage 0
    # scale boxes to image size
    scale_factor = imw / model.input_size
    # shift boxes according to quadrant
    transformed_anchors*= scale_factor
    m, n = np.unravel_index(i, (model.imsplit, model.imsplit))
    transformed_anchors[:, 0] += max(0,n * imw - 2*args.overlap) #n * imw
    transformed_anchors[:, 2] += max(0,n * imw - 2*args.overlap) #n * imw
    transformed_anchors[:, 1] += max(0,m * imh - 2*args.overlap) #m * imh
    transformed_anchors[:, 3] += max(0,m * imh - 2*args.overlap) #m * imh

    return transformed_anchors

def postprocess(model, args, x, imw,imh ,anchors, regression, classification, regressBoxes, clipBoxes):

    threshold =  model.detection_threshold
    iou_threshold = model.nonmax_threshold
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)

    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]

    for i in range(transformed_anchors.shape[0]):
        transformed_anchors[i] = quandrant_box_adjust(model,args,imw,imh, transformed_anchors[i], i % model.imsplit**2)

    s_transformed_anchors = torch.split(transformed_anchors, model.imsplit**2, dim=0)
    s_scores = torch.split(scores, model.imsplit**2, dim=0)
    s_classification = torch.split(classification, model.imsplit**2, dim=0)
    s_scores_over_thresh = torch.split(scores_over_thresh, model.imsplit**2, dim=0)

    tmp_anchors = []
    tmp_scores = []
    tmp_classification = []
    tmp_scores_over_thresh = []
    for k in range(len(s_transformed_anchors)):
        tmp_anchors.append(s_transformed_anchors[k][0])
        tmp_scores.append(s_scores[k][0])
        tmp_classification.append(s_classification[k][0])
        tmp_scores_over_thresh.append(s_scores_over_thresh[k][0])

        for i in range(1,model.imsplit**2):
            tmp_anchors[k] = torch.cat((tmp_anchors[k], s_transformed_anchors[k][i]),dim=0)
            tmp_scores[k] = torch.cat((tmp_scores[k], s_scores[k][i]),dim=0)
            tmp_classification[k] = torch.cat((tmp_classification[k], s_classification[k][i]),dim=0)
            tmp_scores_over_thresh[k] = torch.cat((tmp_scores_over_thresh[k], s_scores_over_thresh[k][i]),dim=0)

    transformed_anchors = torch.stack([fi for fi in tmp_anchors], 0)
    scores = torch.stack([fi for fi in tmp_scores], 0)
    classification = torch.stack([fi for fi in tmp_classification], 0)
    scores_over_thresh = torch.stack([fi for fi in tmp_scores_over_thresh], 0)


    out = []
    #for i in range(x.shape[0]):
    for i in range(transformed_anchors.shape[0]):

        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue


        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]

        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out



def original_postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': boxes_.cpu().numpy(),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs, obj_list, imshow=True, imwrite=False):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        for j in range(len(preds[i]['rois'])):
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            os.makedirs('test/', exist_ok=True)
            cv2.imwrite(f'test/{uuid.uuid4().hex}.jpg', imgs[i])


def replace_w_sync_bn(m):
    for var_name in dir(m):
        target_attr = getattr(m, var_name)
        if type(target_attr) == torch.nn.BatchNorm2d:
            num_features = target_attr.num_features
            eps = target_attr.eps
            momentum = target_attr.momentum
            affine = target_attr.affine

            # get parameters
            running_mean = target_attr.running_mean
            running_var = target_attr.running_var
            if affine:
                weight = target_attr.weight
                bias = target_attr.bias

            setattr(m, var_name,
                    SynchronizedBatchNorm2d(num_features, eps, momentum, affine))

            target_attr = getattr(m, var_name)
            # set parameters
            target_attr.running_mean = running_mean
            target_attr.running_var = running_var
            if affine:
                target_attr.weight = weight
                target_attr.bias = bias

    for var_name, children in m.named_children():
        replace_w_sync_bn(children)


class CustomDataParallel(nn.DataParallel):
    """
    force splitting data to all gpus instead of sending all data to cuda:0 and then moving around.
    """

    def __init__(self, module, num_gpus):
        super().__init__(module)
        #self.num_gpus = num_gpus
        self.num_gpus = 1


    def scatter(self, inputs, kwargs, device_ids):
        # More like scatter and data prep at the same time. The point is we prep the data in such a way
        # that no scatter is necessary, and there's no need to shuffle stuff around different GPUs.
        devices = ['cuda:' + str(x) for x in range(self.num_gpus)]
        splits = inputs[0].shape[0] // self.num_gpus

        if splits == 0:
            raise Exception('Batchsize must be greater than num_gpus.')

        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
                for device_idx in range(len(devices))], \
               [kwargs] * len(devices)

#        return [(inputs[0][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True),
#                 inputs[1][splits * device_idx: splits * (device_idx + 1)].to(f'cuda:{device_idx}', non_blocking=True))
#                for device_idx in range(1)], \
#               [kwargs] * len(devices)




def get_last_weights(weights_path):
    weights_path = glob(weights_path + f'/*.pth')
    weights_path = sorted(weights_path,
                          key=lambda x: int(x.rsplit('_')[-1].rsplit('.')[0]),
                          reverse=True)[0]
    print(f'using weights {weights_path}')
    return weights_path


def init_weights(model):
    for name, module in model.named_modules():
        is_conv_layer = isinstance(module, nn.Conv2d)

        if is_conv_layer:
            if "conv_list" or "header" in name:
                variance_scaling_(module.weight.data)
            else:
                nn.init.kaiming_uniform_(module.weight.data)

            if module.bias is not None:
                if "classifier.header" in name:
                    bias_value = -np.log((1 - 0.01) / 0.01)
                    torch.nn.init.constant_(module.bias, bias_value)
                else:
                    module.bias.data.zero_()


def variance_scaling_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""
    initializer for SeparableConv in Regressor/Classifier
    reference: https://keras.io/zh/initializers/  VarianceScaling
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua','Beige', 'Azure','BlanchedAlmond','Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def from_colorname_to_bgr(color):
    rgb_color=webcolors.name_to_rgb(color)
    result=(rgb_color.blue,rgb_color.green,rgb_color.red)
    return result

def standard_to_bgr(list_color_name):
    standard= []
    for i in range(len(list_color_name)-36): #-36 used to match the len(obj_list)
        standard.append(from_colorname_to_bgr(list_color_name[i]))
    return standard

def get_index_label(label, obj_list):
    index = int(obj_list.index(label))
    return index

def plot_one_box(img, coord, label=None, score=None, color=None, line_thickness=None):
    tl = line_thickness or int(round(0.001 * max(img.shape[0:2])))  # line thickness
    color = color
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 2, 1)  # font thickness
        s_size = cv2.getTextSize(str('{:.0%}'.format(score)),0, fontScale=float(tl) / 3, thickness=tf)[0]
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+s_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}: {:.0%}'.format(label, score), (c1[0],c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)
