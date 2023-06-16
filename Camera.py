import numpy as np
import cv2
import os
# from detection_tools import *
import zmq
import base64
import globalvar
from lib.DisparityProcessor import DisparityProcessor
from zedgetrectifty import init_calibration
from config import *
import sys
from lib.FpsCatcher import FpsCatcher

user = os.environ['HOME']
args = parse_args()

from lib.NetCam import *
import mmap


if args.zed:
    import pyzed.sl as sl


class mmcam:

    def __init__(self,i):
            
        self.dictionary = {
                'width': int(1600*512/800),  #2592, 1920 , 1280 #*512/800
                'height': int(480*512/800), #1520 , 1080 , 720 *512/800
            }

        self.shape = (self.dictionary['height'], self.dictionary['width'], 3)
        self.n = (self.dictionary['height'] * self.dictionary['width'] * 3 * 1)
        self.fd = os.open('/tmp/mmaptest' + str(i), os.O_RDONLY)
        self.mm = mmap.mmap(self.fd, self.n, mmap.MAP_SHARED, mmap.PROT_READ)  # it has to be only for reading

    def read(self):

        self.mm.seek(0)
        buf = self.mm.read(self.n)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(self.shape)
        #        frame = np.frombuffer(buf, dtype=np.float32).reshape(self.shape)

        #        mean=(0.406, 0.456, 0.485)
        #        std=(0.225, 0.224, 0.229)
        #        frame = (frame / 255 - mean) / std
        return frame

    def get(self, setting):
        current_value = self.dictionary[setting]
        return current_value

    def release(self):
        self.mm.close()


class ipcam:

    def __init__(self, ip, port):
        import cv2
        self.cap = cv2.VideoCapture(f'rtsp://admin:m0P04b*PYMG@{ip}:{port}/profile1/media.smp')
        #        self.cap = cv2.VideoCapture(f'rtsp://admin:Admin2021@{ip}:{port}/profile4/media.smp')

        print(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.dictionary = {
            'width': 1280,  # 2592, 1920 , 1280
            'height': 720,  # 1520 , 1080 , 720
        }

    def read(self):
        ret, frame = self.cap.read()
        return frame

    def get(self, setting):
        current_value = self.dictionary[setting]
        return current_value

    def release(self):
        self.cap.release()


class nerian:

    def __init__(self):

        import visiontransfer

        device_enum = visiontransfer.DeviceEnumeration()
        devices = device_enum.discover_devices()
        if len(devices) < 1:
            print('No devices founds')
            sys.exit(1)
        print('Found these devices:')
        for i, info in enumerate(devices):
            print(f'  {i + 1}: {info}')
        selected_device = 0 if len(devices) == 1 else (int(input('Device to open: ') or '1') - 1)
        print(f'Selected device #{selected_device + 1}')
        self.device = devices[selected_device]

        print('Ask parameter server to set stereo mode ...')
        self.params = visiontransfer.DeviceParameters(self.device)

        print('Starting acquisition ...')
        self.transfer = visiontransfer.AsyncTransfer(self.device)

        self.dictionary = {
            'width': self.transfer.collect_received_image_set().get_width() * 2,
            'height': self.transfer.collect_received_image_set().get_height()
        }

    def read(self):
        self.params.set_operation_mode(visiontransfer.OperationMode.RECTIFY)
        image_set = self.transfer.collect_received_image_set()
        left_frame = image_set.get_pixel_data(0, force8bit=True)
        right_frame = image_set.get_pixel_data(1, force8bit=True)

        left_frame = np.stack((left_frame,) * 3, axis=-1)
        right_frame = np.stack((right_frame,) * 3, axis=-1)

        frame = np.concatenate((left_frame, right_frame), axis=1)
        return frame

    def get(self, setting):
        current_value = self.dictionary[setting]
        return current_value

    def get_pointcloud(self):
        self.transfer = visiontransfer.AsyncTransfer(self.device)
        self.params.set_operation_mode(visiontransfer.OperationMode.STEREO_MATCHING)
        image_set = self.transfer.collect_received_image_set()
        rec3d = visiontransfer.Reconstruct3D()
        pointcloud = rec3d.create_point_map(image_set, min_disparity=0.08, max_z=0)
        pointcloud = np.reshape(pointcloud, (self.dictionary['height'], int(self.dictionary['width'] / 2), 3))
        disparity_to_display = image_set.get_pixel_data(1, force8bit=True)

        return pointcloud, disparity_to_display

    def disparity(self):
        self.params.set_operation_mode(visiontransfer.OperationMode.STEREO_MATCHING)
        image_set = self.transfer.collect_received_image_set()
        left_frame = image_set.get_pixel_data(0, force8bit=True)
        disparity = image_set.get_pixel_data(1, force8bit=True)

        return disparity


################################################################################
class Nano_cam:

    def __init__(self, args, linuxid):

        import nanocamera as nano

        if args.resolution == 'NANO':
            width = 800
            height = 480

        if args.resolution == 'HD':
            width = 1280
            height = 720

        self.cap = nano.Camera(device_id=linuxid, flip=2, width=width, height=height, fps=30)

        self.dictionary = {
            'width': self.cap.width,
            'height': self.cap.height,
            'fps': self.cap.fps,
        }

    def read(self):
        frame = self.cap.read()
        return frame

    def get(self, setting):
        current_value = self.dictionary[setting]
        return current_value

    def set(self, setting, new_value):
        self.dictionary[setting] = new_value

    def release(self):
        self.cap.release()


################################################################################


class Opencv_cam:

    def __init__(self, args, linuxid, vflip, hflip):
        isStereo = False
        isCsiCam = False
        if len(args.linuxid) == 1:
            isStereo = True

        if len(args.linuxid) == 2:
            isCsiCam = True

        self.cap = NetCam(capture=args.resolution, source=linuxid, isStereoCam=isStereo,isCsiCam=isCsiCam)
        #self.cap.showDebug()

        if vflip:
            self.cap.invertVertical()
        if hflip:
            self.cap.invertHorizontal()

        self.dictionary = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'hue': cv2.CAP_PROP_HUE,
            'saturation': cv2.CAP_PROP_SATURATION,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'gain': cv2.CAP_PROP_GAIN,
            'whitebalance': cv2.CAP_PROP_WB_TEMPERATURE,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
        }

    def read(self):
        return self.cap.read();

    def get(self, setting):
        return self.cap.get(self.dictionary[setting])

    def set(self, setting, new_values):
        self.cap.set(self.dictionary[setting], new_values)

    def release(self):
        self.cap.release()

    def showDebug(self):
        self.cap.showDebug()


################################################################################

class Zed_cam:

    def __init__(self, args, serial_number, ip):

        depth_mode = args.depth_mode
        min_z = args.min_z
        max_z = args.max_z
        resolution = args.resolution
        vflip = args.vflip
        usb = args.usb

        self.cap = sl.Camera()

        self.dictionary = {
            'brightness': sl.VIDEO_SETTINGS.BRIGHTNESS,
            'contrast': sl.VIDEO_SETTINGS.CONTRAST,
            'hue': sl.VIDEO_SETTINGS.HUE,
            'saturation': sl.VIDEO_SETTINGS.SATURATION,
            'sharpness': sl.VIDEO_SETTINGS.SHARPNESS,
            'gain': sl.VIDEO_SETTINGS.GAIN,
            'exposure': sl.VIDEO_SETTINGS.EXPOSURE,
        }

        init_params = sl.InitParameters()
        #        init_params.set_from_camera_id(i)

        if usb:
            init_params.set_from_serial_number(serial_number)
        else:
            init_params.set_from_stream(ip, args.ports_in[0])  # Specify the IP and port of the sender

        if resolution == 'VGA':
            init_params.camera_resolution = sl.RESOLUTION.VGA
        if resolution == 'HD':
            init_params.camera_resolution = sl.RESOLUTION.HD720
        if resolution == 'FHD':
            init_params.camera_resolution = sl.RESOLUTION.HD1080
        if resolution == '2K':
            init_params.camera_resolution = sl.RESOLUTION.HD2K

        if depth_mode == 'ULTRA':
            init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        if depth_mode == 'QUALITY':
            init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        if depth_mode == 'MEDIUM':
            init_params.depth_mode = sl.DEPTH_MODE.MEDIUM
        if depth_mode == 'PERFORMANCE':
            init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE

        init_params.coordinate_units = sl.UNIT.METER
        init_params.depth_minimum_distance = min_z
        init_params.depth_maximum_distance = max_z
        init_params.sdk_verbose = True
        init_params.camera_fps = 100
        if vflip:
            init_params.camera_image_flip = True
        #        init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

        err = self.cap.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        self.pointcloud = sl.Mat(self.cap.get_camera_information().camera_resolution.width,
                                 self.cap.get_camera_information().camera_resolution.height, sl.MAT_TYPE.F32_C1)
        self.image_left = sl.Mat()
        self.image_right = sl.Mat()

    def read(self):

        self.cap.grab()
        self.cap.retrieve_image(self.image_left, sl.VIEW.LEFT)
        left_frame = self.image_left.get_data()

        self.cap.grab()
        self.cap.retrieve_image(self.image_right, sl.VIEW.RIGHT)
        right_frame = self.image_right.get_data()

        frame = np.concatenate((left_frame, right_frame), axis=1)

        return frame

    def get(self, setting):

        if setting == 'height':
            current_value = self.cap.get_camera_information().camera_resolution.height
        elif setting == 'width':

            current_value = self.cap.get_camera_information().camera_resolution.width * 2
        else:
            current_value = self.cap.get_camera_settings(self.dictionary[setting])

        return current_value

    def set(self, setting, new_values):

        self.cap.set_camera_settings(self.dictionary[setting], new_values)

    def get_pointcloud(self):

        self.cap.retrieve_measure(self.pointcloud, sl.MEASURE.XYZRGBA)
        cloud = self.pointcloud.get_data()
        disparity_to_display = cloud[:, :, 2]
        disparity_to_display[np.isnan(disparity_to_display)] = 0
        disparity_to_display[np.isinf(disparity_to_display)] = 0
        return cloud, disparity_to_display

    def release(self):
        self.cap.close()


################################################################################


################################################################################
class stream_cam:

    def __init__(self, ports_in, ip_address):
        self.cap = NetCam(capture=args.resolution, ip=ip_address, port=ports_in, isStereoCam=True)

        self.dictionary = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'hue': cv2.CAP_PROP_HUE,
            'saturation': cv2.CAP_PROP_SATURATION,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'gain': cv2.CAP_PROP_GAIN,
            'whitebalance': cv2.CAP_PROP_WB_TEMPERATURE,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'width': cv2.CAP_PROP_FRAME_WIDTH,
            'height': cv2.CAP_PROP_FRAME_HEIGHT,
            'fps': cv2.CAP_PROP_FPS,
        }

        self.localDictionary = {
            'brightness': cv2.CAP_PROP_BRIGHTNESS,
            'contrast': cv2.CAP_PROP_CONTRAST,
            'hue': cv2.CAP_PROP_HUE,
            'saturation': cv2.CAP_PROP_SATURATION,
            'sharpness': cv2.CAP_PROP_SHARPNESS,
            'gain': cv2.CAP_PROP_GAIN,
            'whitebalance': cv2.CAP_PROP_WB_TEMPERATURE,
            'exposure': cv2.CAP_PROP_EXPOSURE,
            'width': self.cap.getDetail()['width'],  # imgWidth,
            'height': self.cap.getDetail()['height'],  # imgHeight,
            'fps': cv2.CAP_PROP_FPS,
        }

    def read(self):
        frame = self.cap.read()
        return frame

    def get(self, setting):
        value = self.cap.get(self.dictionary[setting])
        if value is None:
            return self.localDictionary[setting]
        else:
            return value

    def set(self, setting, new_values):
        self.cap.set(self.dictionary[setting], new_values)


    def release(self):
        self.cap.release()

    def showDebug(self):
        self.cap.showDebug()


################################################################################


################################################################################
class Stereocam:

    def __init__(self, args, camleft, camright, serial_number, cam_id):

        from Stereo import Stereo

        self.camleft = camleft
        self.camright = camright

        self.args = args
        self.cam_id = cam_id
        self.cloud_name = 'cloud' + str(cam_id)
        self.cam_name = 'cam' + str(cam_id)

        if camright != None:
            self.width = int(self.camleft.get('width'))
        elif args.ipcam:
            self.width = int(self.camleft.get('width'))

        elif args.mmcam:
            self.width = int(self.camleft.get('width') / 2)

        else:
            self.width = int(self.camleft.get('width') / 2)

        self.height = int(self.camleft.get('height'))

        self.sn = None
        self.port = None
        self.ip = None
        self.image_to_display = None
        self.image_data = None
        self.rigth_image_data = None
        self.netimage_data = None
        self.right_netimage_data = None
        self.cloud = None
        self.disparity_to_display = None
        self.buggycloud = False
        self.distmat = []
        self.detecs = []
        self.object_center_positions = []
        self.object_2d_boxes = []
        self.object_2d_boxes_tracking = []

        self.object_3d_boxes = []
        self.object_dimensions = []
        self.object_2d_centers = [[] for i in range(3)]
        #        self.object_2d_centers = [None,None,None]

        self.speedvec = [[] for i in range(2)]
        self.acccelerationvec = [[] for i in range(1)]
        self.trackingconfidences = []
        self.detectionconfidence = []
        self.nexttrackingposition = []
        self.nexttracking2dbox = []
        self.label = []
        self.trackingids = []
        self.objects = [self.trackingids, self.nexttracking2dbox]

        self.disparityProcessor = None
        self.stereo = None
        self.rectifydown=args.rectifydown

        basePath = os.path.dirname(os.path.realpath(__file__))
        configFileName = f'{basePath}/calibration/SN{serial_number}.json'
        if not os.path.exists(configFileName):
            configFileName = f'{basePath}/zedinfo/SN{serial_number}.conf'

        print(f'Loading camera config file : {configFileName}')

        settings = {}
        settings['resolution'] = args.resolution
        settings['configFileName'] = configFileName
        settings['imageSize'] = [self.width, self.height]
        self.disparityProcessor = DisparityProcessor(settings)

        # self.stereo = Stereo(args)
        # # get calibration coefficients
        # calibration_path = os.path.dirname(os.path.realpath(__file__)) + '/zedinfo/'
        # calibration_file = calibration_path + 'SN' + str(serial_number) + '.conf'
        # camera_matrix_left, camera_matrix_right, map_left_x, map_left_y, map_right_x, map_right_y, Tvec, Rvec, distCoeffs_left, distCoeffs_right, R1, \
        # px_left, py_left, px_right, fx, fy,Q = \
        #     init_calibration(calibration_file, self.width, self.height, args.resolution)
        # f = (fx + fy) / 2
        # Tvec = Tvec.flatten()
        # Tvec = Tvec / 1000
        # base = -Tvec[0]
        # Q[3,2]=Q[3,2]*1000
        #
        # self.coeff = [map_left_x, map_left_y, map_right_x, map_right_y, f, base, px_left, py_left,
        #               Q, distCoeffs_left, Tvec, Rvec, R1, camera_matrix_left]

    def read(self):
        left_frame = self.camleft.read()
        left_frame = cv2.flip(left_frame, 1)
        left_frame = cv2.flip(left_frame, 0)
        if self.camright != None:
            right_frame = self.camright.read()
            frame = np.concatenate((left_frame, right_frame), axis=1)
        else:
            frame = left_frame
        return frame

    def get(self, camera_settings):
        return self.camleft.get(camera_settings)

    def set(self, camera_settings, new_values):
        self.camleft.set(camera_settings, new_values)
        if self.camright != None:
            self.camright.set(camera_settings, new_values)

    def get_pointcloud(self, left_frame_rect, right_frame_rect, displayDisparity=True):

        disparity = None
        depthMap = None
        disparity_to_display = None

        if globalvar.stereoFps is None:
            globalvar.stereoFps = FpsCatcher(autoStart=False)
        now = FpsCatcher.currentMilliTime()

        if self.args.zed:
            cloud, disparity_to_display = self.camleft.get_pointcloud()
        elif self.args.opencv:
            cloud, disparity, depthMap = self.disparityProcessor.computeDisparityMap(left_frame_rect, right_frame_rect)

            # cloud,disparity_to_display = self.stereo.get_pointcloud(left_frame_rect, right_frame_rect, self.coeff, self.cloud_name)
        elif self.args.nerian:
            cloud, disparity_to_display = self.camleft.get_pointcloud()
        elif self.args.mmcam:
            cloud, disparity, depthMap = self.disparityProcessor.computeDisparityMap(left_frame_rect, right_frame_rect)
            # cloud,disparity_to_display = self.stereo.get_pointcloud(left_frame_rect, right_frame_rect, self.coeff, self.cloud_name)
        elif self.args.nanocam:
            cloud, disparity, depthMap = self.disparityProcessor.computeDisparityMap(left_frame_rect, right_frame_rect)
            # cloud,disparity_to_display = self.stereo.get_pointcloud(left_frame_rect, right_frame_rect, self.coeff, self.cloud_name)

        if displayDisparity and disparity is not None:
            #  Compute the disparityTo dispay only when needed
            #disparity_to_display = self.disparityProcessor.convertDisparityToColor(disparity)
            disparity_to_display = self.disparityProcessor.convertDisparityToGray(disparity)
        current = FpsCatcher.currentMilliTime()
        globalvar.stereoFps.compute(current - now)

        return cloud, disparity_to_display

    def initalize_stereo_trackbar(self):
        # TODO : self.stereo is not used anymore since DisparityProcessor replace it
        #  So right now there is no trackbar anymore -> Re add them if needed
        if self.stereo is not None:
            self.stereo.initalize_stereo_trackbar(self.width, self.height, self.cloud_name)

    def get_rectified_left_right(self):

        # [map_left_x, map_left_y, map_right_x, map_right_y] = self.coeff[0:4]

        if self.camright != None:
            left_frame = self.camleft.read()
            right_frame = self.camright.read()
            self.netimage_data = left_frame
            self.right_netimage_data = right_frame
        else:
            frame = self.camleft.read()
            frames = np.split(frame, 2, axis=1)
            [left_frame, right_frame] = frames
            self.netimage_data = left_frame
            self.right_netimage_data = right_frame


            
        if args.nerian:
            left_frame_rect = left_frame
            right_frame_rect = right_frame
            self.netimage_data = left_frame
            self.right_netimage_data = right_frame

        elif args.ipcam:
            frame = self.camleft.read()
            left_frame_rect = frame
            right_frame_rect = frame

        elif args.mmcam:
            #            frame = self.camleft.read()
            left_frame_rect = left_frame
            right_frame_rect = right_frame
        #            left_frame_rect = cv2.remap(left_frame[:, :, 0:3], map_left_x, map_left_y, interpolation=cv2.INTER_LINEAR)
        #            right_frame_rect = cv2.remap(right_frame[:, :, 0:3], map_right_x, map_right_y, interpolation=cv2.INTER_LINEAR)

                                  
        elif args.zed:
            frame = self.camleft.read()
            left_frame_rect = left_frame
            right_frame_rect = right_frame
        else:
            left_frame_rect, right_frame_rect = self.disparityProcessor.rectifyLeftRight(left_frame, right_frame)

        return left_frame_rect, right_frame_rect

    def net_get_rectified_left(self):
        #frame = self.camleft.read()
        # if args.stereo:
        #     frames = np.split(frame, 2, axis=1)
        #     [left_frame, right_frame] = frames
        # else:
        #     left_frame = frame
        #     right_frame = None
        #frames = np.split(frame, 2, axis=1)
        #[left_frame, right_frame] = frames

        #netleft_frame_rect  = left_frame if args.nerian else self.disparityProcessor.netrectifyLeft(left_frame)
        netleft_frame_rect  =  self.netimage_data #if args.nerian else self.disparityProcessor.netrectifyLeft(self.netimage_data)

        return netleft_frame_rect

    def get_left_right(self):

        if self.camright != None:
            left_frame = self.camleft.read()
            right_frame = self.camright.read()
        elif args.ipcam:
            frame = self.camleft.read()
            left_frame = frame
            right_frame = frame
        else:
            frame = self.camleft.read()
            frames = np.split(frame, 2, axis=1)
            [left_frame, right_frame] = frames

        return left_frame, right_frame


    def release(self):
        self.camleft.release()
        if self.camright != None:
            self.camright.release()

    def showDebug(self):
        self.camleft.showDebug()


################################################################################
