################################################################################
##          Classe DisparityProcessor
##      Gere le calcul des cartes de disparite
##  Author : J. Coupez
##  Date : 17/02/2021
##  Copyright Dista
##  Version : 1.0
################################################################################
import json
import cv2
import numpy
from lib.Calibration import Calibration
from lib.RectifiedCamera import RectifiedCamera
from zedgetrectifty import init_calibration
from config import *
args = parse_args()

class DisparityProcessor:
    DEFAULT_WINDOW_NAME = 'Disparity Map'
    PANEL_FONTFACE = cv2.FONT_HERSHEY_DUPLEX
    PANEL_FONTSCALE = 1.5
    PANEL_FONTTHICKNESS = 2
    PANEL_TEXT_COLOR = (255, 255, 255)

    def __init__(self, settings={}):
        self.configFileName = settings.get('configFileName') or 1
        self.resolution = settings.get('resolution') or 'FHD'
        self.imageSize = settings.get('imageSize') or None

        # Camera Rectification
        self.rectifiedLeft = None
        self.rectifiedRight = None
        self.NETrectifiedLeft = None
        self.disparityMap = None
        self.depthMap = None
        self.calibration = self.getCoefficientFromConfigFile()

        # Disparity Params
        self.disparityDownScale = 2
        self.usefastBilateral = True

        # CREATION STEREO MATCHERS -------------------------------------------------
        #self.num_disp = 5 * 16
        self.num_disp = int(10 * 16 / (self.disparityDownScale * args.rectifydown))
        min_disp = 1
        blockSize = 5
        #self.left_matcher = cv2.StereoBM_create(numDisparities=self.num_disp, blockSize=blockSize)
        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=self.num_disp,
            blockSize=blockSize,
            P1=24 * blockSize * blockSize,
            P2=96 * blockSize * blockSize,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
        # --------------------------------------------------------------------------

        # CREATE WLS FILTER --------------------------------------------------------
        lambda_wls = 8000.0
        sigma_wls = 1.5
        self.wlsFilter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
        self.wlsFilter.setLambda(lambda_wls)
        self.wlsFilter.setSigmaColor(sigma_wls)
        # --------------------------------------------------------------------------

        # Create rectified Camera
        if self.calibration and self.rectifiedLeft is None:
            self.initRectifiedCamera()

    def getCoefficientFromConfigFile(self):
        """
        Retieve the settings form a camera JSON config file
        See a JSON sample in calibration folder
        @return:
        """

        if self.configFileName.endswith(".json"):
            with open(self.configFileName, 'r') as json_file:
                jsonCoef = json.load(json_file)
                coefForResolution = jsonCoef.get(self.resolution) or None
                if coefForResolution is not None:
                    return Calibration(coefForResolution)
        else:
            # Old file format
            width, height = self.imageSize[0], self.imageSize[1]
            cameraMatrix_left, cameraMatrix_right, map_left_x, map_left_y, map_right_x, map_right_y,netmap_left_x, netmap_left_y,T,R,distCoeffs_left, distCoeffs_right, R1 ,R2,left_cam_cx,left_cam_cy,right_cam_cx,left_cam_fx,left_cam_fy,Q = \
                init_calibration(self.configFileName, width, height, self.resolution)
            # Adjust scale
            T = T.flatten()
            T = T / 1000
            Q[3, 2] = Q[3, 2] * 1000
            settings = {}
            settings['imageSize'] = self.imageSize
            settings['cameraMatrixLeft'] = cameraMatrix_left.tolist()
            settings['cameraMatrixRight'] = cameraMatrix_right.tolist()
            settings['distCoefLeft'] = distCoeffs_left.tolist()
            settings['distCoefRight'] = distCoeffs_right.tolist()
            settings['T'] = T.tolist()
            settings['R'] = R.tolist()

            self.rectifiedLeft = RectifiedCamera(cameraMatrix_left, distCoeffs_left, R1, cameraMatrix_left, map_left_x, map_left_y, Q)
            self.rectifiedRight = RectifiedCamera(cameraMatrix_right, distCoeffs_right, R2, cameraMatrix_right, map_right_x, map_right_y, Q)
            self.net_rectifiedLeft = RectifiedCamera(cameraMatrix_left, distCoeffs_left, R1, cameraMatrix_left, netmap_left_x, netmap_left_y, Q)
            return Calibration(settings)
        return None

    def initRectifiedCamera(self):
        """
        Rectify a stereo Camera with intrinsic & extrinsic correction.
        This method don't have to be called manually as it's called during DisparityProcessor initialisation
        """
        # Getting Camera configuration
        cameraMatrix1 = numpy.array(self.calibration.getCameraMatrixLeft())
        cameraMatrix2 = numpy.array(self.calibration.getCameraMatrixRight())
        distCoeffs1 = numpy.array(self.calibration.getDistCoefLeft())
        distCoeffs2 = numpy.array(self.calibration.getDistCoefRight())
        T = numpy.array(self.calibration.getT())
        R = numpy.array(self.calibration.getR())
        imageSize = self.calibration.getImageSize()
        height, width = imageSize[1], imageSize[0]
        downscale=args.rectifydown  # to downscale rectified image

        # Computes rectification transforms for each head of a calibrated stereo camera.
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1=cameraMatrix1, cameraMatrix2=cameraMatrix2,
                                                    distCoeffs1=distCoeffs1, distCoeffs2=distCoeffs2,
                                                    R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
                                                    imageSize=(width, height), newImageSize=(int(width / downscale), int(height/ downscale)))

        # Computes the undistortion and rectification transformation map for each camera
        map_left_x, map_left_y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (int(width / downscale), int(height/ downscale)),
                                                             cv2.CV_32FC1)
        map_right_x, map_right_y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (int(width / downscale), int(height/ downscale)),
                                                               cv2.CV_32FC1)
        # Store each rectification for later use
        self.rectifiedLeft = RectifiedCamera(cameraMatrix1, distCoeffs1, R1, P1, map_left_x, map_left_y, Q)
        self.rectifiedRight = RectifiedCamera(cameraMatrix2, distCoeffs2, R2, P2, map_right_x, map_right_y, Q)


        # Neural net image : computes the undistortion and rectification transformation map for neural net image (size is original)
        # Computes rectification transforms for each head of a calibrated stereo camera.
        netR1, netR2, netP1, netP2, netQ, _, _ = cv2.stereoRectify(cameraMatrix1=cameraMatrix1, cameraMatrix2=cameraMatrix2,
                                                    distCoeffs1=distCoeffs1, distCoeffs2=distCoeffs2,
                                                    R=R, T=T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
                                                    imageSize=(width, height), newImageSize=(width, height))

        netmap_left_x, netmap_left_y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, netR1, netP1, (width, height),
                                                             cv2.CV_32FC1)
        # Store each rectification for later use
        self.net_rectifiedLeft = RectifiedCamera(cameraMatrix1, distCoeffs1, netR1, netP1, netmap_left_x, netmap_left_y, netQ)

    def rectifyLeftRight(self, imageLeft, imageRight):
        if not self.calibration:
            return [], []
        # Undistort each image
        imageRectifiedLeft = self.rectifiedLeft.rectifyImage(imageLeft)
        imageRectifiedRight = self.rectifiedRight.rectifyImage(imageRight)
        return imageRectifiedLeft, imageRectifiedRight


    def netrectifyLeft(self, imageLeft):
        if not self.calibration:
            return [], []
        # Undistort each image
        netimageRectifiedLeft = self.net_rectifiedLeft.rectifyImage(imageLeft)
        return netimageRectifiedLeft


    def computeDisparityMap(self, imageRectifiedLeft, imageRectifiedRight):
        """
Compute the disparity Map from a left & right picture took from a camera
        @param imageLeft: raw left image
        @param imageRight: raw right image
        @return: disparityMap, depthMap
        """
        if not self.calibration:
            return [], []

        width = imageRectifiedLeft.shape[1]
        height = imageRectifiedLeft.shape[0]

        # Reduce the size of each picture
        reducedWidth = int(width / self.disparityDownScale)
        reducedHeight = int(height / self.disparityDownScale)
        reducedImageLeft = cv2.resize(imageRectifiedLeft, (reducedWidth, reducedHeight))
        reducedImageRight = cv2.resize(imageRectifiedRight, (reducedWidth, reducedHeight))

        # Convert image to grayscale
        grayedImageLeft = cv2.cvtColor(reducedImageLeft, cv2.COLOR_RGB2GRAY)
        grayedImageRight = cv2.cvtColor(reducedImageRight, cv2.COLOR_RGB2GRAY)

        # Apply a blur to smooth big difference in a smal area ( due to noise )
        grayedImageLeft = cv2.medianBlur(grayedImageLeft, 3)
        grayedImageRight = cv2.medianBlur(grayedImageRight, 3)

        # Computes disparity map for the specified stereo pair
        disparityLeft = self.left_matcher.compute(cv2.UMat(grayedImageLeft), cv2.UMat(grayedImageRight))
        disparityRight = self.right_matcher.compute(cv2.UMat(grayedImageRight), cv2.UMat(grayedImageLeft))
        disparityLeft = numpy.int16(cv2.UMat.get(disparityLeft))
        disparityRight = numpy.int16(cv2.UMat.get(disparityRight))

        # Apply filtering to the disparity map.
        disparityMap = self.wlsFilter.filter(disparityLeft, imageRectifiedLeft, None, disparityRight)

        # Dividing disparity map by 16 to be used with reprojectImageTo3D
        disparityMap = disparityMap.astype(numpy.float32) / 16.0

        # Get the confidence map that was used in the last filter call.
        confidenceMap = self.wlsFilter.getConfidenceMap()
        roi = numpy.array(self.wlsFilter.getROI()) * self.disparityDownScale
        mask = numpy.zeros(confidenceMap.shape, confidenceMap.dtype)
        mask[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]] = 1

        # Apply a Bilateral filter ( but need OpenCv to be compiled with needs to be compiled with EIGEN)
#        try:
#            if self.usefastBilateral:
#                fbs_spatial = 8.0
#                fbs_luma = 8.0
#                fbs_chroma = 8.0
#                fbs_lambda = 128.0
#                solved_disparityMap = cv2.ximgproc.fastBilateralSolverFilter(imageRectifiedLeft, disparityMap,
#                                                                             confidenceMap / 255.0, None,
#                                                                             fbs_spatial, fbs_luma, fbs_chroma,
#                                                                             fbs_lambda)
#                disparityMap = solved_disparityMap.astype(numpy.float32) / 16.0 * mask
#        except:
#            print(
#                "EIGEN not installed, Disabling fastBilateral filter. see https://eigen.tuxfamily.org/dox/GettingStarted.html")
#            self.usefastBilateral = False

        # Reprojects the disparity image to 3D space.
        cloud = cv2.reprojectImageTo3D(disparityMap, self.rectifiedLeft.Q, handleMissingValues=True)
        depthMap = cloud[:, :, 2] * mask

        return cloud, disparityMap, depthMap

    def convertDisparityToColor(self, disparityMap):
        """
Transform a disparityMap into a visual RGB colored picture
        @param disparityMap: A disparityMap computed by computeDisparityMap
        @return: a RGB color map with the size of the original picture
        """
        # disparityMap = numpy.clip(disparityMap,0,200)
        # print(numpy.amax(disparityMap))
        disparityMap = cv2.normalize(disparityMap, disparityMap, 0, 256,
                                     cv2.NORM_MINMAX, cv2.CV_8UC3)
        disparityMap = cv2.applyColorMap(disparityMap, cv2.COLORMAP_JET)
        return disparityMap

    def convertDisparityToGray(selfself, disparityMap):
        """
Transform a disparityMap into a visual grayscale picture
        @param disparityMap: A disparityMap computed by computeDisparityMap
        @return: a grayscale picture with the size of the original picture
        """
        if len(disparityMap) == 0:
            return None
        minDisp = numpy.min(disparityMap)
        maxDisp = numpy.max(disparityMap)
        disparity_to_display = ((disparityMap - minDisp) / (maxDisp - minDisp) * 255.).astype(numpy.uint8)
        return disparity_to_display

    def getPointDepth(self, depthMap, point2D):
        """
Retrieve the specific Depth in meter
        @param depthMap:
        @param point2D:
        @return:
        """
        if not self.calibration:
            return 0
        x, y = point2D
        try:
            depth = depthMap[int(y)][int(x)]
        except IndexError as e:
            depth = 0
        
        return depth


    def release(self):
        del self.wlsFilter
        del self.left_matcher
        del self.right_matcher
        del self.calibration

