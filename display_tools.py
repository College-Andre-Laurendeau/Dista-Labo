import os

user = os.environ['HOME']
import numpy as np
import cv2
from config import *
from scipy.spatial import distance_matrix
import mmap
import time
import errno


args = parse_args()

if not args.show :
    from vidgear.gears import NetGear
    from vidgear.gears.helper import reducer

################################################################################
def get_fonts_and_padding_from_resolution(resolution_str):
    fontscale = 1
    if resolution_str == 'VGA':
        fontscale = 0.6
    elif resolution_str == 'HD':
        fontscale = 1
    elif resolution_str == 'FHD':
        fontscale = 1.3
    elif resolution_str == '2K':
        fontscale = 1.5

    return fontscale


################################################################################


################################################################################
def set_windows(args):
    window_verti_position = args.window_verti_position
    window_hori_position = args.window_hori_position
    windowName = 'Camera Live Feed'
    if args.fullscreen:
        cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL)  # cv2.WINDOW_NORMAL
        cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        w = args.window_width
        h = args.window_height
        cv2.namedWindow(windowName, cv2.WINDOW_GUI_NORMAL)  # cv2.WINDOW_NORMAL
        cv2.resizeWindow(windowName, w, h)
        cv2.moveWindow(windowName, window_hori_position, window_verti_position)


def TO_DELETE_set_windows(args, cap, i):
    window_verti_position = args.window_verti_position
    window_hori_position = args.window_hori_position
    w = cap.width
    h = cap.height
    cloud_name = cap.cloud_name
    cam_name = cap.cam_name

    camgrid = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2],
               [2, 3], [2, 4]]

    if args.fullscreen:

        cv2.namedWindow(cam_name, cv2.WINDOW_GUI_NORMAL)  # cv2.WINDOW_NORMAL
        cv2.namedWindow(cam_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(cam_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    else:

        #        window_width=w
        #        window_height=h
        #        print(w)
        #        print(h)
        w = args.window_width
        h = args.window_height

        cv2.namedWindow(cam_name, cv2.WINDOW_GUI_NORMAL)  # cv2.WINDOW_NORMAL

        if args.showcloud:
            cv2.namedWindow(cloud_name, cv2.WINDOW_GUI_NORMAL)  # cv2.WINDOW_NORMAL

        cv2.resizeWindow(cam_name, w, h)
        if args.showcloud:
            cv2.resizeWindow(cloud_name, w, h)

        cv2.moveWindow(cam_name, 0 + window_hori_position + camgrid[i][0] * w,
                       window_verti_position + camgrid[i][1] * h)

    ################################################################################


def display_image_with_stereo_options(args, cap, cloud):
    cam_name = cap.cam_name
    image_data = cap.image_data
    right_image_data = cap.right_image_data
    w = cap.width
    h = cap.height

    if args.sidebyside:
        image = np.concatenate((image_data, right_image_data), axis=1)
        cv2.resizeWindow(cam_name, w, h)

    else:
        image = np.copy(image_data)

    image[-20:-1, -20:-1, :] = 0

    cv2.imshow(cam_name, image)


################################################################################


################################################################################
def display_image(args, i, cap):
    if args.mirror == True:
        cap.image_data = np.copy(cv2.flip(cap.image_data, 1))

    cv2.imshow(cap.cam_name, cap.image_data)


################################################################################


################################################################################
def draw_2d_box(cap, bounding_box_2d, obj_list, fontscale, k):
    #if cap.detectionconfidence[k] >= 0.4:
    color = (255, 255, 0)
    #else:
    #    color = (0, 0, 255)
    cv2.rectangle(cap.image_to_display, (int(bounding_box_2d[0, 0]), int(bounding_box_2d[0, 1])),
                  (int(bounding_box_2d[2, 0]), int(bounding_box_2d[2, 1])),
                  color, 2)  # 3

    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1 * fontscale
    font_thickness = 1  # 1
    if args.tracking:
        text_str = 'ID :' + str(cap.trackingids[k])
    else:
        text_str = obj_list[cap.label[k]]

    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
    text_pt = (int(bounding_box_2d[0, 0]), int(bounding_box_2d[0, 1]) - text_h)
    cv2.putText(cap.image_to_display, text_str, text_pt, font_face, font_scale, (0, 0, 255), font_thickness,
                cv2.LINE_AA)
    ################################################################################


################################################################################
def display_distances_on_image(image_data, bounding_box_2d, position, fontscale):
    font_face = cv2.FONT_HERSHEY_DUPLEX
    font_scale = fontscale * 1
    font_thickness = 2
    #                     coins N-O et S-E de la fenêtre 2D
    upperleftcorner = (int(bounding_box_2d[0, 0]), int(bounding_box_2d[0, 1]))
    lowerrightcorner = (int(bounding_box_2d[2, 0]), int(bounding_box_2d[2, 1]))
    lowerleftcorner = (int(bounding_box_2d[3, 0]), int(bounding_box_2d[3, 1]))
    # milieu de la fenètre 2D
    #middle_row = int(np.round((upperleftcorner[0] + lowerrightcorner[0]) / 2))
    #middle_col = int(np.round((upperleftcorner[1] + lowerrightcorner[1]) / 2))
    #text_pt4 = (middle_row, middle_col)

    text_str4 = ' %.2fm' % np.around(
        np.sqrt(np.power(position[0], 2) + np.power(position[1], 2) + np.power(position[2], 2)), 2)

    text_w, text_h = cv2.getTextSize(text_str4, font_face, font_scale, font_thickness)[0]
    text_pt4 = (int(bounding_box_2d[3, 0]), int(bounding_box_2d[3, 1]) +  int(text_h))

    cv2.putText(image_data, text_str4, text_pt4, font_face, font_scale, (0, 0, 255), font_thickness)  # cv2.LINE_AA


################################################################################


##########################################################################################################

def write_detections_on_image(args, cap, obj_cat_id, obj_list):
    
    for i in range(len(cap)):
#            print('WRITE')
#            print(len(cap[i].object_center_positions))
#            print(len(cap[i].object_2d_boxes))
        
#            coeff = cap[i].coeff
#            w = cap[i].width
#            h = cap[i].height
#        
#            [map_left_x, map_left_y, map_right_x, map_right_y, f, base, px_left, py_left, Q, distCoeffs_left, Tvec, Rvec, R1,
#             camera_matrix_left] = coeff
#            [k1, k2, p1, p2, k3] = distCoeffs_left
            fontscale = get_fonts_and_padding_from_resolution(args.resolution)
        
        
            ####################################################################                              
            # LOOP OVER ALL DETECTIONS FOR DISPLAY 
            for k in range(0, min(len(cap[i].object_2d_boxes), args.max_number_objects)):  # len(id_colors)
        
        
                label = cap[i].label[k]  # int(detecs[k,7])
                bounding_box_2d = cap[i].object_2d_boxes[k]
        
                if args.justdetec:
                    ####################################################################
                    # DRAW 2D BOUNDING BOX
                    if args.mirror == True:
                        bounding_box_2d[:, 0] = w - bounding_box_2d[:, 0]
                    if args.view_2dbox:
                        draw_2d_box(cap[i], bounding_box_2d, obj_list, fontscale, k)
                    #                          if len(cap[i].nexttracking2dbox)!=0:
                    #                              for i in range(len(cap[i].nexttracking2dbox)):
                    #                                  draw_2d_box(cap[i],cap[i].nexttracking2dbox[i],obj_list,fontscale,k)
                    #
        
        
                else:

#    
#                    print('k',k)
                    position = np.copy(cap[i].object_center_positions[k])
                    if args.view_3dbox : dimensions = np.copy(cap[i].object_dimensions[k])
                    if args.view_3dbox : bounding_box = np.copy(cap[i].object_3d_boxes[k])
                    
                    #if not np.isnan(np.linalg.norm(position)):
        
                    ####################################################################
                    # DRAW 2D BOUNDING BOX
                    if args.mirror == True:
                        bounding_box_2d[:, 0] = w - bounding_box_2d[:, 0]

                    if args.view_2dbox:
                        draw_2d_box(cap[i], bounding_box_2d, obj_list, fontscale, k)
                    #                          if len(cap[i].nexttracking2dbox)!=0:
                    #                              for i in range(len(cap[i].nexttracking2dbox)):
                    #                                  draw_2d_box(cap[i],cap[i].nexttracking2dbox[i],obj_list,fontscale,k)
            
                        ####################################################################
                        # DISPLAY DISTANCES AND POSITION ON IMAGE AND SIDE PANEL
                        if args.displaydistance:
                            display_distances_on_image(cap[i].image_to_display, bounding_box_2d, position, fontscale)
            
                        ####################################################################
                        # DRAW 3D BOX : IF OBJECT NOT TOO CLOSE (1 meter) AND TOO SMALL
                        if args.view_3dbox and bounding_box.shape[0] == 8 and dimensions[0] * dimensions[1] * dimensions[
                            2] > 0.2 and position[2] > 1:
                            imgpts = map_3d_to_2d(args, cap[i], bounding_box)
                            display_3d_bounding_box(cap[i].image_to_display, imgpts)
            
                            ####################################################################            
                        # DRAW SEPARATING ARROW
        
                if len(cap[i].object_center_positions) > 1:
        
                    if args.arrows:
                        display_arrows(cap[i], args.lineupmode, args.lineupdepth, args.maxdist, \
                                       fontscale, args.red_box_arrow_yellow_box, args.mirror, args.nb_neighbors,
                                       args.minheight, args.minwidth, args.mindepth)
        
                            ####################################################################


################################################################################    
def map_3d_to_2d(args, cap, points):
    [map_left_x, map_left_y, map_right_x, map_right_y, f, base, px_left, py_left, Q, distCoeffs_left, Tvec, Rvec, R1,
     camera_matrix_left] = cap.coeff
    [k1, k2, p1, p2, k3] = distCoeffs_left

    imgpts = np.zeros((len(points), 2)).astype(int)

    for j in range(len(points)):

        point = np.matmul(R1, points[j])
        x = point[0]
        y = point[1]
        z = point[2]
        xp = x / z
        yp = y / z
        r = np.sqrt(np.power(xp, 2) + np.power(yp, 2))
        xpp = xp * (1 + k1 * np.power(r, 2) + k2 * np.power(r, 4) + k3 * np.power(r, 6)) + 2 * p1 * xp * yp + p2 * (
                np.power(r, 2) + 2 * np.power(xp, 2))
        ypp = yp * (1 + k1 * np.power(r, 2) + k2 * np.power(r, 4) + k3 * np.power(r, 6)) + p1 * (
                np.power(r, 2) + 2 * np.power(yp, 2)) + 2 * p2 * xp * yp

        u = int(np.round(xpp * f + px_left))
        v = int(np.round(ypp * f + py_left))

        if args.mirror == True:
            imgpts[j, 0] = cap.widht - u
        else:
            imgpts[j, 0] = u
        imgpts[j, 1] = v

    return imgpts



################################################################################
def display_3d_bounding_box(image_data, imgpts):
    thick = 12
    cv2.line(image_data, tuple(imgpts[0].ravel()), tuple(imgpts[1].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[3].ravel()), tuple(imgpts[0].ravel()), (255, 255, 0), thick)

    cv2.line(image_data, tuple(imgpts[0].ravel()), tuple(imgpts[0 + 4].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[1].ravel()), tuple(imgpts[1 + 4].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[2].ravel()), tuple(imgpts[2 + 4].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[3].ravel()), tuple(imgpts[3 + 4].ravel()), (255, 255, 0), thick)

    cv2.line(image_data, tuple(imgpts[0 + 4].ravel()), tuple(imgpts[1 + 4].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[1 + 4].ravel()), tuple(imgpts[2 + 4].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[2 + 4].ravel()), tuple(imgpts[3 + 4].ravel()), (255, 255, 0), thick)
    cv2.line(image_data, tuple(imgpts[3 + 4].ravel()), tuple(imgpts[0 + 4].ravel()), (255, 255, 0), thick)


################################################################################    


################################################################################
def display_arrows(cap, lineupmode, lineupdepth, maxdist, \
                   fontscale, red_box_arrow_yellow_box, mirror, nb_neighbors, minheight, minwidth, mindepth):
    image_data = cap.image_to_display
    distmat = cap.distmat
    imgpts = cap.object_2d_centers

    # LOOP OVER ALL DETECTION/PERSONS

    for i in range(0, distmat.shape[0]):
        
        if not np.isnan(np.linalg.norm(cap.object_center_positions[i])):

            if lineupmode:
                nb_neighbors = 2
                dist = np.copy(distmat[i])
                sortdist = np.argsort(dist)
                sortdist = sortdist[1:min(1 + nb_neighbors, len(sortdist))]
                lineupdepth = lineupdepth
            else:
                dist = np.copy(distmat[i])
                sortdist = np.argsort(dist)
                sortdist = sortdist[1:min(1 + nb_neighbors, len(sortdist))]
                lineupdepth = maxdist
    
            # LOOP OVER ALL NEIGHBOORS
    
            for j in range(0, len(sortdist)):  # if draw to closest neighboors only

                target_id = sortdist[j]
                #obj_h = cap.object_dimensions[i][1]
                #target_h = cap.object_dimensions[target_id][1]
                # obj_w = cap.object_dimensions[i][0]
                # target_w = cap.object_dimensions[target_id][0]
                # obj_d = cap.object_dimensions[i][2]
                # target_d = cap.object_dimensions[target_id][2]
                obj_z = cap.object_center_positions[i][2]
                target_z = cap.object_center_positions[target_id][2]

                if distmat[i, target_id] < 1.95 and distmat[i, target_id] < maxdist : #and obj_z < maxdepth and target_z < maxdepth :#and obj_h > minheight and target_h > minheight and obj_w > minwidth and target_w > minwidth and obj_d > mindepth and target_d > mindepth:
                    midarrow = imgpts[i] + (imgpts[target_id] - imgpts[i]) / 2
                    text_str = str(np.round(distmat[i, target_id], 2))
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    thick = 3
                    font_scale = fontscale
                    font_thickness = 1
                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    text_pt = (int(midarrow[0]), int(midarrow[1] + text_h+15))

                    cv2.putText(image_data, text_str, text_pt, font_face, font_scale, [0, 0, 255], font_thickness,
                                cv2.LINE_AA)
                    cv2.line(image_data, tuple(imgpts[i].ravel()), tuple(imgpts[target_id].ravel()), [0, 0, 255],
                             int(thick))
    
                if distmat[i, target_id] >= 2 and distmat[i, target_id] < maxdist : # and obj_z < maxdepth and target_z < maxdepth :#and obj_h > minheight and target_h > minheight and obj_w > minwidth and target_w > minwidth and obj_d > mindepth and target_d > mindepth:
                    midarrow = imgpts[i] + (imgpts[target_id] - imgpts[i]) / 2
                    text_str = str(np.round(distmat[i, target_id], 2))
                    font_face = cv2.FONT_HERSHEY_DUPLEX
                    font_scale = fontscale
                    font_thickness = 1
                    text_w, text_h = cv2.getTextSize(text_str, font_face, font_scale, font_thickness)[0]
                    thick = 3
                    text_pt = (int(midarrow[0]), int(midarrow[1] + text_h+15))

                    cv2.putText(image_data, text_str, text_pt, font_face, font_scale, [0, 255, 255], font_thickness,
                                cv2.LINE_AA)
                    cv2.line(image_data, tuple(imgpts[i].ravel()), tuple(imgpts[target_id].ravel()), [0, 255, 255],
                             int(thick))


def display_and_record(args, capList):
    finalPicture = None
    frameLine = None
    camPerLine = 1
    camCount = len(capList)
    if camCount > 3:
        camPerLine = int(np.sqrt(camCount)) +1

    lineIndex = 0
    imgHeight = 0
    imgWidth = 0
    
    windowName = 'Camera Live Feed'
    resizefactor =  1/4
#    w = args.window_width
#    h = args.window_height
#    cv2.resizeWindow(windowName, (int(w*resizefactor), int(h*resizefactor)))

    #  Scan all pictures
    for i in range(camCount):
        #if not args.mmcam and not args.nanocam:
        #    capList[i].showDebug()
            # Display the picture with the box (computation result)
        image = capList[i].image_to_display

        if image is None:
            continue
        # imgHeight, imgWidth, _ = image.shape
        imgWidth = int(args.window_width)
        imgHeight = int(args.window_height)

        image = cv2.resize(image, (int(imgWidth / camPerLine), int(imgHeight / camPerLine)))
            

        lineIndex = lineIndex + 1
        if frameLine is None:
            frameLine = image
        else:
            frameLine = np.concatenate((frameLine, image), axis=1)
            if lineIndex >= camPerLine:
                #  the line is full : append to previously created
                lineIndex = 0
                if finalPicture is None:
                    finalPicture = frameLine
                else:
                    finalPicture = np.concatenate((finalPicture, frameLine), axis=0)
                frameLine = None

    #  Add the last pictures
    if frameLine is not None:
        if camPerLine - lineIndex > 0:
            blankImage = np.zeros( (int(imgHeight / camPerLine), int(imgWidth / camPerLine) * (camPerLine - lineIndex), 3), dtype=np.uint8)
            frameLine = np.concatenate((frameLine, blankImage), axis=1)
        if finalPicture is None:
            # There was only 1 line of picture
            finalPicture = frameLine
        else:
            finalPicture = np.concatenate((finalPicture, frameLine), axis=0)

    if finalPicture is None:
        return


#    finalPicture=cv2.resize(finalPicture,(int(finalPicture.shape[1]*resizefactor),
#                                          int(finalPicture.shape[0]*resizefactor)))

    
    cv2.waitKey(1)
    cv2.imshow(windowName, finalPicture)
    #cv2.imshow(windowName, capList[0].disparity_to_display)




def display_avg_obj_count(frame, displayWidth, avg):
    debugTextSize = displayWidth / 1280
    thickness = 1 if displayWidth < 1280 else 2
    TEXT_COLOR = (0, 0, 255)
    TEXT_POSITION = (0, 0)
    textPosX, textPosY = TEXT_POSITION
    textPosX += int(40 * debugTextSize)
    textPosY += int(40 * debugTextSize)
    cv2.putText(frame, f'Max person count in last 5 frames : {avg} ',
                (textPosX, textPosY),
                cv2.FONT_HERSHEY_SIMPLEX, debugTextSize, TEXT_COLOR, thickness,
                cv2.LINE_AA)


def memfile(args,camCount):

    
    camPerLine = 1
#    if camCount > 3:
    camPerLine = int(np.sqrt(camCount)) +1    
    
    resizefactor = 1 if args.show else 0.1

    
    w = int(args.window_width  * resizefactor * camPerLine)
    h = int(args.window_height  * resizefactor)
    n = (h*w*3)
    
    fd = os.open('/tmp/img_mmap', os.O_CREAT | os.O_TRUNC | os.O_RDWR)
    os.truncate(fd, n)  # resize file
    
    return [fd,n]




def img_pipe():
    
#    FIFO = '/var/run/mypipe'
    FIFO = '/home/dista/Npipe.mpjg'
    try:
        os.mkfifo(FIFO)
    except OSError as oe: 
        if oe.errno != errno.EEXIST:
            raise

#    os.mkfifo(FIFO)
    
    return FIFO













