import os

user = os.environ['HOME']
from config import *
from zedgetrectifty import init_calibration
import numpy as np
from Camera import *
import numpy as np
import cv2
import mmap
from display_tools import *
from config import *

args = parse_args()

########################################


if len(args.linuxid) != 0:
    args.linuxid = list(map(int, args.linuxid.split(',')))

if len(args.ports_in) != 0:
    args.ports_in = list(map(int, args.ports_in.split(',')))

if len(args.ports_out) != 0:
    args.ports_out = list(map(int, args.ports_out.split(',')))

if len(args.vflip) != 0:
    args.vflip = list(map(int, args.vflip.split(',')))

if len(args.hflip) != 0:
    args.hflip = list(map(int, args.hflip.split(',')))

if len(args.pwd) != 0:
    pwd = list(args.pwd.split(','))

if len(args.camid) != 0:
    camid = list(args.camid.split(','))

################################################################################
def initalize_all_cameras(serial_numbers, ip_addresses, cam_ports):
    cap = {}
    socket_stream = {}

    for i in range(len(serial_numbers)):

        serial_number = serial_numbers[i]



        if args.usb:

            if args.nanocam:

                if len(args.linuxid) == 1:

                    cap[i] = Stereocam(args, Nano_cam(args.inuxid[0]), None, serial_number, i)


                elif len(args.linuxid) == 2:

                    cap[i] = Stereocam(args, Nano_cam(args, args.linuxid[0]), Nano_cam(args, args.linuxid[1]),
                                       serial_number, i)


            elif args.opencv:

                if len(args.linuxid) == 1:

                    cap[i] = Stereocam(args, Opencv_cam(args, args.linuxid[0], vflip=args.vflip, hflip=args.hflip),
                                       None, serial_number, i)

                elif len(args.linuxid) == 2:
                    vFlipLeft = False
                    vFlipRight = False
                    hFlipLeft = False
                    hFlipRight = False

                    if len(args.vflip) > 0:
                        vFlipLeft = args.vflip[0]
                        vFlipRight = args.vflip[1]

                    if len(args.hflip) > 0:
                        hFlipLeft = args.hflip[0]
                        hFlipRight = args.hflip[1]
                    # print(args.vflip[1])

                    cap[i] = Stereocam(args, Opencv_cam(args, args.linuxid[0], vflip=vFlipLeft, hflip=hFlipLeft),
                                       Opencv_cam(args, args.linuxid[1], vflip=vFlipRight, hflip=hFlipRight),
                                       serial_number, i)

                #                cap[i].set('fps',100)


            elif args.zed:

                if len(args.ip) != 0:
                    ip = args.ip[i]
                else:
                    ip = []
                cap[i] = Stereocam(args, Zed_cam(args, serial_number, ip), None, serial_number, i)



            elif args.nerian:

                cap[i] = Stereocam(args, nerian(), None, serial_number, i)

            elif args.mmcam:

                cap[i] = Stereocam(args, mmcam(camid[i]), None, serial_number, i)



        elif not args.usb:

            if args.ipcam:
                cap[i] = Stereocam(args, ipcam(ip_addresses[i], cam_ports[i],pwd[i]), None, serial_number, i)

            else:
                cap[i] = Stereocam(args, stream_cam(cam_ports[i], ip_addresses[i]), None, serial_number, i)

        cap[i].ip = None
        if len(ip_addresses) > i:
            cap[i].ip = ip_addresses[i]

        cap[i].port = None
        if len(cam_ports) > i:
            cap[i].port = cam_ports[i]

        cap[i].sn = serial_number
    return cap


################################################################################


################################################################################
opencv_cam_settings = "brightness"
str_camera_settings = "brightness"
step_camera_settings = 1


################################################################################
def opencv_adjust(usb, cap):

    key = 1000
    while key != 113:  # for 'q' key

        frame = cap.read()

        if frame is not None:
            frames = np.split(frame, 2, axis=1)
            left_frame = frames[0]
            cv2.imshow(cap.cam_name, left_frame)
            key = cv2.waitKey(1)
            if key != -1:
                opencv_camera_settings(key, cap)

    return


################################################################################


################################################################################
def print_opencv_camera_information(cam):
    print("Resolution: {0}, {1}.".format(cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))


################################################################################
def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Quit:                               q\n")


################################################################################


opencv_to_v4l2_dict = {
    cv2.CAP_PROP_BRIGHTNESS: 'brightness',
    cv2.CAP_PROP_CONTRAST: 'contrast',
    cv2.CAP_PROP_HUE: 'hue',
    cv2.CAP_PROP_SATURATION: 'saturation',
    cv2.CAP_PROP_SHARPNESS: 'sharpness',
    cv2.CAP_PROP_GAIN: 'gain',
    cv2.CAP_PROP_EXPOSURE: 'brightness',
    cv2.CAP_PROP_WB_TEMPERATURE: 'gain'
}


################################################################################
def opencv_camera_settings(key, cap):
    step_camera_settings = 1

    if key == 115:  # for 's' key
        switch_opencv_camera_settings()
    elif key == 43:  # for '+' key
        current_value = cap.get(opencv_cam_settings)
        cap.set(opencv_cam_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cap.get(opencv_cam_settings)
        if current_value >= 1:
            cap.set(opencv_cam_settings, current_value - step_camera_settings)

            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cap.set('brightness', -1)
        cap.set('contrast', -1)
        cap.set('hue', -1)
        cap.set('saturation', -1)
        cap.set('gain', -1)
        cap.set('whitebalance', -1)
        cap.set('exposure', -1)

        print("Camera settings: reset")


################################################################################
def switch_opencv_camera_settings():
    global opencv_cam_settings
    global str_camera_settings
    if opencv_cam_settings == 'brightness':
        opencv_cam_settings = 'contrast'
        str_camera_settings = "contrast"
        print("Camera settings: CONTRAST")
    elif opencv_cam_settings == "contrast":
        opencv_cam_settings = "hue"
        str_camera_settings = "hue"
        print("Camera settings: HUE")
    elif opencv_cam_settings == "hue":
        opencv_cam_settings = 'saturation'
        str_camera_settings = 'saturation'
        print("Camera settings: SATURATION")
    elif opencv_cam_settings == 'saturation':
        opencv_cam_settings = 'sharpness'
        str_camera_settings = "sharpness"
        print("Camera settings: Sharpness")
    elif opencv_cam_settings == 'sharpness':
        opencv_cam_settings = 'gain'
        str_camera_settings = "gain"
        print("Camera settings: GAIN")
    elif opencv_cam_settings == 'gain':
        opencv_cam_settings = 'exposure'
        str_camera_settings = "exposure"
        print("Camera settings: EXPOSURE")
    elif opencv_cam_settings == 'exposure':
        opencv_cam_settings = 'whitebalance'
        str_camera_settings = "whitebalance"
        print("Camera settings: WHITEBALANCE")
    elif opencv_cam_settings == 'whitebalance':
        opencv_cam_settings = 'brightness'
        str_camera_settings = "brightness"
        print("Camera settings: BRIGHTNESS")
################################################################################
