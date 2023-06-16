################################################################################
##          Classe Calibrator
##      Gere la calibration de caméra (stéréo ou non)
##  Author : J. Coupez
##  Date : 10/10/2020
##  Copyright Dista
##  Version : 1.0
################################################################################

import sys
import os
import cv2
import numpy as np
import json
import glob
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from threading import Thread

from lib.NetCam import NetCam
from lib.CountDown import CountDown
from lib.CalibrationMonoProcessor import CalibrationMonoProcessor
from lib.CalibrationStereoProcessor import CalibrationStereoProcessor
from lib.DisparityProcessor import DisparityProcessor
from lib.DisparityDisplay import DisparityDisplay


class Calibrator:
    DEFAULT_WINDOW_NAME = 'Calibration'
    DISPARITY_WINDOW_NAME = 'DisparityMap'

    BOARD_MAX_CORNER = (20, 20)
    BOARD_MIN_CORNER = (3, 3)
    BOARD_SHAPE = None

    # Panel Constants
    PANEL_TEXT_COLOR = (255, 255, 255)
    PANEL_LINE_COLOR = (200, 200, 200)

    PANEL_FONTFACE = cv2.FONT_HERSHEY_DUPLEX
    PANEL_FONTSCALE = 0.8
    PANEL_FONTTHICKNESS = 1

    def __init__(self, settings={}):

        self.cameraIndexes = settings.get('cameraIndexes') or 0
        self.fishEye = settings.get('fishEye') or False
        self.squareSize = settings.get('squareSize') or 1
        self.serial = settings.get('serial') or 'TEST'
        self.resolution = settings.get('resolution') or 'FHD'
        self.isCsiCam = settings.get('isCsiCam') or False
        boardShape = settings.get('boardShape') or None
        if boardShape is not None and boardShape != '':
            Calibrator.BOARD_SHAPE = (boardShape[0], boardShape[1])

        self.filePath = f'calibration/{self.serial}'
        self.clearPreviousPicture = settings.get('clearPreviousPicture')

        self.boardColShape, self.boardRowShape = Calibrator.BOARD_MAX_CORNER

        # Left Camera
        self.cameraLeft = None
        self.imageLeft = None
        self.cameraLeftSetting = settings.get('leftCamera') or {}
        self.leftCorners = []
        self.isAllLeftCorners = False
        self.leftCalibrationProcessor = CalibrationMonoProcessor('left', self.filePath, self.resolution,
                                                                 Calibrator.getChessBoardForFrame)

        # Right Camera
        self.cameraRight = None
        self.imageRight = None
        self.cameraRightSetting = settings.get('rightCamera') or {}
        self.rightCorners = []
        self.isAllRightCorners = False
        self.rightCalibrationProcessor = CalibrationMonoProcessor('right', self.filePath, self.resolution,
                                                                  Calibrator.getChessBoardForFrame)

        ## Stereo
        self.isStereo = False
        self.stereoCalibrationProcessor = CalibrationStereoProcessor('stereo',
                                                                     self.filePath,
                                                                     self.resolution,
                                                                     Calibrator.getChessBoardForFrame,
                                                                     leftCalibrationProcessor=self.leftCalibrationProcessor,
                                                                     rightCalibrationProcessor=self.rightCalibrationProcessor)

        self.displayScaleFactor = 0.7

        # Computing Properties
        self.activeProcessor = None  # Store the current running calibration processor
        self.captureType = None
        self.photoCountDown = CountDown(5)
        self.lastCornerPosition = None
        self.lastPhotoCornerPosition = None
        self.currentCalibrationIndex = None  # Store the index of the selected calibration ( for preview)
        self.currentImageIndex = -1
        self.previewCalibration = False

        # Drawing properties
        self.isDrawingOverLay = False
        self.lineList = []
        self.lastPoint = None
        self.mousePosition = None
        self.testBascule = False

        # Disparity properties
        # self.disparityCursorPosition = None
        self.isShowDisparity = False
        self.disparityProcessor = None
        self.disparityDisplay = None

        # Autorotation stepper
        self.stepperCurrentRotation = 0
        self.autoShoot = False

    def run(self):
        self.saveConfiguration()

        self.createFolder()
        if self.clearPreviousPicture:
            self.clearFolder()

        self.leftCalibrationProcessor.resetImageIndex()
        self.rightCalibrationProcessor.resetImageIndex()
        self.stereoCalibrationProcessor.resetImageIndex()

        # Init camera Left
        if len(self.cameraIndexes) > 0:
            self.cameraLeft = self.initCamera(0, self.cameraLeftSetting)

        # If there is a right camera, same thing
        if len(self.cameraIndexes) > 1:
            self.cameraRight = self.initCamera(1, self.cameraRightSetting)

        if self.cameraLeft is None:
            return

        # Setup the stereo properly
        if self.cameraLeft.isStereoCam or self.cameraRight:
            self.isStereo = True
            if self.cameraLeft.isStereoCam:
                # Camera with only 1 USB cable and is a steroCam
                # replace the right Camera for the left
                self.cameraRight = self.cameraLeft

        # If no Chessboard shape was provided : Detect the board size
        while Calibrator.BOARD_SHAPE is None:
            if self.cameraLeft is not None and self.cameraLeft.isRunning():
                finalPicture = self.cameraLeft.readLeft()
                lowQualityFrame = cv2.resize(finalPicture, (800, 600))
                Calibrator.BOARD_SHAPE = self.detectBoardShape(lowQualityFrame)
                self.renderBoardDetection(finalPicture)
                # Grab user input if needed
                self.listenKeyboard()
            else:
                return

        # Now we have a chessboard : Launch all picture processors
        self.launchProcessor()

        while self.cameraLeft is not None and self.cameraLeft.isRunning():
            displayLeft = None
            displayRight = None

            # Read picture and find corners
            if self.captureType == "left" or self.captureType == 'stereo':
                self.imageLeft = self.cameraLeft.readLeft()
                self.isAllLeftCorners, self.leftCorners, _ = Calibrator.getChessBoardForFrame(self.imageLeft)
                displayLeft = self.imageLeft.copy()
                displayLeft = self.drawChessBoard(displayLeft, self.leftCorners, self.isAllLeftCorners)
            if self.captureType == "right" or self.captureType == 'stereo':
                self.imageRight = self.cameraRight.readRight()
                self.isAllRightCorners, self.rightCorners, _ = Calibrator.getChessBoardForFrame(self.imageRight)
                displayRight = self.imageRight.copy()
                displayRight = self.drawChessBoard(displayRight, self.rightCorners, self.isAllRightCorners)

            if self.currentImageIndex == -1 or self.activeProcessor is None:
                # Run the auto photo mode
                finalPicture = self.photoAutoShooter(displayLeft, displayRight, self.imageLeft, self.imageRight)
            else:
                # Take the picture from the index
                finalPicture = self.activeProcessor.getImage(self.currentImageIndex)
                if self.captureType == "stereo":
                    displayLeft = self.activeProcessor.getImage(self.currentImageIndex, 'left')
                    displayRight = self.activeProcessor.getImage(self.currentImageIndex, 'right')
                else:
                    displayLeft = finalPicture
                    displayRight = finalPicture

            # Add the undistorted picture previewer
            if self.previewCalibration:
                finalPicture = self.previewCalibrationResult(finalPicture, displayLeft, displayRight)

            # Add the drawingLayer
            finalPicture = self.drawOverlay(finalPicture)

            # Add the footer to the display
            finalPicture = self.renderFooter(finalPicture)

            self.render(finalPicture)

            if self.isShowDisparity:
                self.renderDisparityMap(self.imageLeft, self.imageRight)

            # Grab user input if needed
            self.listenKeyboard()

    ######### INIT & CLEANUP METHODS #################
    def initCamera(self, cameraIndex, cameraSetting):
        source = self.cameraIndexes[cameraIndex]
        isStereoCam = True
        if len(self.cameraIndexes) > 1:
            isStereoCam = False
        camera = NetCam(source=source, capture=self.resolution, isStereoCam=isStereoCam, isCsiCam=self.isCsiCam)
        if cameraSetting.get('vFlip'):
            camera.invertVertical()
        if cameraSetting.get('hFlip'):
            camera.invertHorizontal()
        return camera

    def release(self):
        if self.cameraLeft:
            self.cameraLeft.release()
            self.cameraLeft = None
        if self.cameraRight:
            self.cameraRight.release()
            self.cameraRight = None

    def createFolder(self):
        if not os.path.exists('calibration'):
            os.makedirs('calibration')

        path = self.filePath
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(f'{path}/left'):
            os.makedirs(f'{path}/left')
        if not os.path.exists(f'{path}/right'):
            os.makedirs(f'{path}/right')
        if not os.path.exists(f'{path}/stereo'):
            os.makedirs(f'{path}/stereo')

    def clearFolder(self):
        # Old picture cleanup

        path = self.filePath

        if self.captureType == "left":
            path = path + "/left/*.jpg"
        elif self.captureType == "right":
            path = path + "/right/*.jpg"
            self.rightCalibrationProcessor.resetImageIndex()
        elif self.captureType == "stereo":
            path = path + "/stereo/*.jpg"
            self.stereoCalibrationProcessor.resetImageIndex()
        else:
            # Delete all pictures
            path = path + "/**/*.jpg"
            self.leftCalibrationProcessor.resetImageIndex()
            self.rightCalibrationProcessor.resetImageIndex()
            self.stereoCalibrationProcessor.resetImageIndex()

        # Flush the content of the directory
        files = glob.glob(path, recursive=True)
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

        print(f'\tCleared {len(files)} picture(s) in {path}...')

    def saveConfiguration(self):
        # Save the calibration for the next run
        with open('calibrate_settings.json', 'w') as outfile:
            json.dump({'cameraIndexes': self.cameraIndexes,
                       'serial': self.serial,
                       'fishEye': self.fishEye,
                       'squareSize': self.squareSize,
                       'resolution': self.resolution,
                       'isCsiCam': self.isCsiCam,
                       'boardShape': Calibrator.BOARD_SHAPE or '',
                       'leftCamera': {
                           'vFlip': self.cameraLeftSetting.get('vFlip') or False,
                           'hFlip': self.cameraLeftSetting.get('hFlip') or False,
                       },
                       'rightCamera': {
                           'vFlip': self.cameraRightSetting.get('vFlip') or False,
                           'hFlip': self.cameraRightSetting.get('hFlip') or False,
                       },
                       }, outfile, indent=4)

    def saveCalibration(self):
        jsonConfigurationFile = None
        if self.currentCalibrationIndex is not None:
            calibration = self.stereoCalibrationProcessor.getCoefficient(self.currentCalibrationIndex)
            if calibration:
                jsonConfigurationFile = f'calibration/{self.serial}.json'
                with open(jsonConfigurationFile, 'w') as outfile:
                    json.dump({f'{self.resolution}': calibration.toJSON()}, outfile, indent=4)
                    print(f'Saved config to calibration/{self.serial}.conf')
                with open(f'zedinfo/{self.serial}.conf', 'a') as outfile:
                    outfile.write(calibration.toTxt())
                    outfile.close()
                    print(f'Saved config to zedinfo/{self.serial}.conf')
        return jsonConfigurationFile

    def openDisparityWindow(self):
        if self.isShowDisparity == False:
            jsonConfigurationFile = self.saveCalibration()
            if jsonConfigurationFile is not None and self.imageLeft is not None:
                settings = {}
                settings['resolution'] = self.resolution
                settings['configFileName'] = jsonConfigurationFile
                height, width, _ = self.imageLeft.shape
                settings['imageSize'] = [width, height]
                print(jsonConfigurationFile)

                self.disparityProcessor = DisparityProcessor(settings=settings)
                self.disparityDisplay = DisparityDisplay(self.disparityProcessor)
                self.isShowDisparity = True

    def closeDisparityWindow(self):
        if self.isShowDisparity == True:
            self.isShowDisparity = False
            if self.disparityProcessor is not None:
                self.disparityProcessor.release()
            if self.disparityDisplay is not None:
                self.disparityDisplay.release()

            self.disparityProcessor = None
            self.disparityDisplay = None

    ########## PICTURE GRABBING METHODS ################
    def photoAutoShooter(self, imageLeft, imageRight, rawLeft, rawRight):
        if self.captureType is None:
            return

        countDownFrame = None
        currentCornerPosition = None
        savePicture = False

        if self.captureType == 'left':
            countDownFrame = imageLeft
            if self.isAllLeftCorners:
                currentCornerPosition = self.leftCorners[0][0]
        elif self.captureType == 'right':
            countDownFrame = imageRight
            if self.isAllRightCorners:
                currentCornerPosition = self.rightCorners[0][0]
        elif self.captureType == 'stereo':
            countDownFrame = np.concatenate((imageLeft, imageRight), axis=1)
            if self.isAllLeftCorners and self.isAllRightCorners:
                currentCornerPosition = self.leftCorners[0][0]

        if self.lastCornerPosition is None:
            self.lastCornerPosition = currentCornerPosition

        if self.autoShoot and currentCornerPosition is not None:
            if self.isInsideBoundingBox(self.lastCornerPosition, currentCornerPosition, 10):
                # the first corner is near the currentCorner
                if not self.isInsideBoundingBox(self.lastPhotoCornerPosition, currentCornerPosition, 20):
                    if self.photoCountDown.isFinished:
                        # We take only 1 picture in the same area (corner must have moved by a square of 40 pixels)
                        savePicture = True
                        self.lastPhotoCornerPosition = currentCornerPosition
                    elif not self.photoCountDown.isRunning:
                        self.photoCountDown.start()
            else:
                # The corner has moved since last time : Reinit the countdown
                self.photoCountDown.stop()
                self.lastCornerPosition = None
                self.lastPhotoCornerPosition = None
        else:
            # No Corner found : stop the countdown
            self.photoCountDown.stop()

        if countDownFrame is not None:
            if self.photoCountDown.isRunning:
                ### Display countDown
                remainingTime = self.photoCountDown.getRemainingSeconds()
                if (remainingTime < 4):
                    height, width, _ = countDownFrame.shape
                    countDownPosition = (int(width / 2), int(height - 150))
                    countDownFrame = cv2.circle(countDownFrame, countDownPosition, 150, (255, 255, 255), -1)
                    countDownFrame = cv2.circle(countDownFrame, countDownPosition, 150, (0, 0, 0), 3)
                    countDownFrame = cv2.putText(countDownFrame, f'{remainingTime}',
                                                 (countDownPosition[0] - 80, countDownPosition[1] + 80),
                                                 cv2.FONT_HERSHEY_SIMPLEX,
                                                 8.0, (0, 0, 0), 10)
            elif self.lastPhotoCornerPosition is not None:
                ### Display photoshoot
                pt1 = (0, 0)
                height, width, _ = countDownFrame.shape
                pt2 = (width, height)
                overlay = countDownFrame.copy()
                overlay = cv2.rectangle(overlay, pt1, pt2, (255, 255, 255), -1)
                countDownFrame = cv2.addWeighted(overlay, 0.4, countDownFrame, 1 - 0.4, 0, countDownFrame)
                countDownFrame = cv2.putText(countDownFrame, f'Image "{self.captureType}" prise, deplacez le damier...',
                                             (int(width / 2 - 220), int(height / 2 + 40)),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        if savePicture:
            if self.captureType == 'left':
                self.leftCalibrationProcessor.savePicture(rawLeft)
            elif self.captureType == 'right':
                self.rightCalibrationProcessor.savePicture(rawRight)
            elif self.captureType == 'stereo':
                self.stereoCalibrationProcessor.savePicture(rawLeft, rawRight)

        finalPicture = countDownFrame

        return finalPicture

    def photoManual(self):
        if self.captureType == 'left':
            self.leftCalibrationProcessor.savePicture(self.imageLeft)
        elif self.captureType == 'right':
            self.rightCalibrationProcessor.savePicture(self.imageRight)
        elif self.captureType == 'stereo':
            self.stereoCalibrationProcessor.savePicture(self.imageLeft, self.imageRight)

    ########## RENDERING METHODS ###########
    def render(self, finalPicture):
        # Display the final output
        height, width, _ = finalPicture.shape
        width = int(width * self.displayScaleFactor)
        height = int(height * self.displayScaleFactor)
        finalPicture = cv2.resize(finalPicture, (width, height), interpolation=cv2.INTER_LINEAR)

        cv2.imshow(Calibrator.DEFAULT_WINDOW_NAME, finalPicture)
        cv2.setMouseCallback(Calibrator.DEFAULT_WINDOW_NAME, self.clickAndPoint)

    def renderBoardDetection(self, finalPicture):
        height, width, _ = finalPicture.shape
        footerHeight = 100
        footerPanel = np.zeros(shape=(footerHeight, width, 3), dtype=np.uint8)
        self.drawText(footerPanel,
                      f'Detection de la taille du damier...{(self.boardColShape, self.boardRowShape)}',
                      x=0, y=0, width=width, height=footerHeight, alignH='center', alignV='center',
                      fontScale=0.8)
        finalPicture = np.concatenate((finalPicture, footerPanel), axis=0)
        self.render(finalPicture)

    def renderFooter(self, finalPicture):

        height, width, _ = finalPicture.shape

        footerHeight = 400
        footerWidth = width
        sidePanelWidth = 450
        if (2 * sidePanelWidth > footerWidth):
            sidePanelWidth = int(footerWidth / 3)
        centerPanelWidth = footerWidth - 2 * sidePanelWidth

        leftPanel = np.zeros(shape=(footerHeight, sidePanelWidth, 3), dtype=np.uint8)
        rightPanel = np.zeros(shape=(footerHeight, sidePanelWidth, 3), dtype=np.uint8)
        centerPanel = np.zeros(shape=(footerHeight, centerPanelWidth, 3), dtype=np.uint8)

        # Draw Center Panel
        cv2.rectangle(centerPanel, (0, 0), (centerPanelWidth, footerHeight), Calibrator.PANEL_LINE_COLOR, 2)
        self.renderFooterLeft(leftPanel, sidePanelWidth, footerHeight)
        self.renderFooterCenter(centerPanel)
        self.renderFooterRight(rightPanel, sidePanelWidth, footerHeight)

        # Merge all side panel as a footer
        footerFrame = np.concatenate((leftPanel, centerPanel, rightPanel), axis=1)
        # Merge the footer with the main frame
        finalPicture = np.concatenate((finalPicture, footerFrame), axis=0)

        return finalPicture

    def renderFooterLeft(self, panel, sidePanelWidth, footerHeight):
        # Draw the right Panel
        if self.activeProcessor is None:
            return
        lastY = 5
        cv2.rectangle(panel, (0, 0), (sidePanelWidth, footerHeight), Calibrator.PANEL_LINE_COLOR, 2)

        if self.currentImageIndex == -1:
            cv2.rectangle(panel, (0, lastY - 2), (sidePanelWidth, lastY + 26), (0, 0, 200), -1)
        lastY = self.drawText(panel, f'<Live Stream {self.captureType}>', 5, lastY)
        imageList = self.activeProcessor.chessboardImageList
        startIndex = 0
        rms = ""
        if self.currentImageIndex > 5:
            startIndex = self.currentImageIndex - 5
        for i in range(startIndex, len(imageList)):
            captureImage = imageList[i]
            if type(captureImage) is tuple:
                fileName = captureImage[0].getShortFileName() + " " + captureImage[1].getShortFileName()
            else:
                fileName = captureImage.getShortFileName()
                rms = captureImage.getRmsErrorAsString()
            if i == self.currentImageIndex:
                # Highlight the current line
                cv2.rectangle(panel, (0, lastY - 2), (sidePanelWidth, lastY + 26), (90, 90, 90), -1)
            if self.activeProcessor.isImageDisabled(i):
                cv2.rectangle(panel, (0, lastY + 10), (sidePanelWidth, lastY + 10), (255, 255, 255), 2)

            lastY = self.drawText(panel, f'{fileName} {rms}', 15, lastY)
            if lastY > footerHeight:
                break

    def renderFooterCenter(self, centerPanel):
        height, width, _ = centerPanel.shape
        if self.activeProcessor is None or len(self.activeProcessor.cameraCoeficientList) == 0:
            self.drawText(centerPanel,
                          f'En attente de calibration...',
                          x=0, y=0, width=width, height=height, alignH='center', alignV='center',
                          fontScale=0.8)
            return

        # Draw Per view RMS on the left
        fig = Figure()
        fig.set_figheight(height / 100.0)
        fig.set_figwidth(width / 200.0)
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        x = []
        y = []
        cameraCoeficient = self.activeProcessor.getCoefficient(self.currentCalibrationIndex)
        if cameraCoeficient is not None:
            rms = cameraCoeficient.getRms()
            x.append(0)
            y.append(rms)
            perErrorList = cameraCoeficient.getPerViewErrors()
            fullImageListLen = len(self.activeProcessor.chessboardImageList)
            perViewIndex = 0
            for i in range(fullImageListLen):
                x.append(i + 1)
                value = rms
                if not self.activeProcessor.isImageDisabled(i) and len(perErrorList) > perViewIndex:
                    # As the full list can have some removed picture, the PerViewError array may not have the same size
                    # We can find the right rms for this picture by incrementing only when the picture was not disabled
                    value = sum(perErrorList[perViewIndex]) / len(perErrorList[perViewIndex])
                    perViewIndex += 1  # only increment the index when the index has been used for calibration

                y.append(value)

            ax.plot(x, y, label='Per image RMS')  # Plot more data on the axes...
            ax.set_xlabel('Images')  # Add an x-label to the axes.
            ax.set_ylabel('RMS')  # Add a y-label to the axes.
            ax.set_title(f'Erreur par image (RMS = {rms})')  # Add a title to the axes.
            ax.fill_between(x, rms, y)
            ax.axvline(self.currentImageIndex + 1, 0, 1, label='Current Image', c='g', linewidth=3.0)
            ax.legend()  # Add a legend.
            ax.grid(True)  # lines

            canvas.draw()  # draw the canvas, cache the renderer
            buf = canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            image = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            imHeight, imWidth, _ = image.shape
            xdecal = int(width / 2)
            centerPanel[0:image.shape[0], 0:image.shape[1]] = image

        if self.currentCalibrationIndex is not None:
            # Draw calibration result List on the right
            fig = Figure()
            fig.set_figheight(height / 100.0)
            fig.set_figwidth(width / 200.0)
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            x = []
            y = []
            for i in range(len(self.activeProcessor.cameraCoeficientList)):
                cameraCoeficient = self.activeProcessor.cameraCoeficientList[i]
                x.append(i + 1)
                y.append(cameraCoeficient.getRms())

            ax.plot(x, y, label='RMS', c='r', linewidth=3)  # Plot more data on the axes...
            ax.set_xlabel('Calibrations')  # Add an x-label to the axes.
            ax.set_ylabel('RMS')  # Add a y-label to the axes.
            ax.set_title(
                f'Calibrations Selectionnée : {self.currentCalibrationIndex} - {len(perErrorList)} images')  # Add a title to the axes.
            ax.grid(True)  # draw grid lines
            ax.axvline(self.currentCalibrationIndex + 1, 0, 1, label='Current Calibration', c='b', linewidth=3.0)
            ax.legend()  # Add a legend.

            canvas.draw()  # draw the canvas, cache the renderer
            buf = canvas.tostring_rgb()
            ncols, nrows = fig.canvas.get_width_height()
            image = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
            imHeight, imWidth, _ = image.shape
            centerPanel[0:image.shape[0], xdecal:xdecal + image.shape[1]] = image

    def renderFooterRight(self, panel, sidePanelWidth, footerHeight):
        # Draw the right Panel ( key map)
        cv2.rectangle(panel, (0, 0), (sidePanelWidth, footerHeight), Calibrator.PANEL_LINE_COLOR, 2)
        lastY = 5
        # lastY = self.drawText(leftPanel, f'Controles :', 5, 5)

        if self.captureType is not None:
            if self.isStereo:
                # lastY = self.drawText(panel, f'Touches : (Camera {self.captureType})', 5, lastY + 5)
                lastY = self.drawText(panel, f'ENTER : cam suivante ', 10, lastY)
            lastY = self.drawText(panel, f'SPACE : Photo ', 10, lastY)
            if not self.captureType == "stereo":
                lastY = self.drawText(panel, f'h: (H) flip', 10, lastY)
                lastY = self.drawText(panel, f'v: (V) flip', 10, lastY)

            if self.isStereo and not self.cameraLeft.isStereoCam:
                lastY = self.drawText(panel, f'i : (i)nverse camera ', 20, lastY)
                cv2.rectangle(panel, (0, lastY), (sidePanelWidth, footerHeight), Calibrator.PANEL_LINE_COLOR, 2)

            lastY = self.drawText(panel, f'w/s: Previous/Next picture', 10, lastY)
            lastY = self.drawText(panel, f'p: (P)review undistorted', 10, lastY)
            lastY = self.drawText(panel, f'a/d: Previous/Next calibration', 10, lastY)

            lastY = self.drawText(panel, f't: au(t)o shoot (current : {self.autoShoot})', 10, lastY)
            lastY = self.drawText(panel, f'r: (R)emove current picture', 10, lastY)
            if self.captureType == "stereo":
                lastY = self.drawText(panel, f'm: show Disparity(M)ap', 10, lastY)
                lastY = self.drawText(panel, f'c: save (c)onfig file', 10, lastY)
                # lastY = self.drawText(panel, f'b: (b)ascule {self.testBascule}', 10, lastY)

            lastY = self.drawText(panel, f'x: Drawing tools', 10, lastY)
        self.drawText(panel, f'ESC : Quitter', 10, lastY)

    def previewCalibrationResult(self, finalPicture, imageLeft, imageRight):
        if self.activeProcessor is None or self.currentCalibrationIndex is None:
            return finalPicture

        undistorted = None
        if self.captureType == 'left':
            undistorted = self.activeProcessor.undistort(imageLeft, self.currentCalibrationIndex)
        elif self.captureType == 'right':
            undistorted = self.activeProcessor.undistort(imageRight, self.currentCalibrationIndex)
        elif self.captureType == 'stereo':
            return finalPicture

        if undistorted is not None:
            height, width, _ = undistorted.shape
            # TODO : recrire proprement les textes
            cv2.rectangle(undistorted, (0, 0), (350, 30), Calibrator.PANEL_LINE_COLOR, -1)
            self.drawText(undistorted,
                          f'Image apres correction',
                          x=10, y=5, width=width, height=height, fontScale=0.8, color=(0, 0, 0))
            if self.currentImageIndex >= 0:
                # Draw the chessboard onto the picture
                chessboardImage = self.activeProcessor.getChessboardImage(self.currentImageIndex)
                if chessboardImage is not None and len(chessboardImage.distortedCorners) > 0:
                    finalPicture = self.drawChessBoard(chessboardImage.getImage().copy(),
                                                       chessboardImage.getDistortedCorners(),
                                                       chessboardImage.getIsAllCorners())
            finalPicture = np.concatenate((finalPicture, undistorted), axis=1)

        return finalPicture

    def drawOverlay(self, finalPicture):
        height, width, _ = finalPicture.shape
        if self.isDrawingOverLay:
            for pointTuple in self.lineList:
                x1, y1 = pointTuple[0]
                x2, y2 = pointTuple[1]

                cv2.line(finalPicture, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2, lineType=8)

            height, width, _ = finalPicture.shape
            if self.mousePosition is not None:
                x, y = self.mousePosition
                if x > width:
                    x = width
                if y > height:
                    y = height
                x = int(x)
                y = int(y)
                cv2.line(finalPicture, (0, y), (width, y), (255, 0, 0), thickness=1, lineType=8)
                cv2.line(finalPicture, (x, 0), (x, height), (255, 0, 0), thickness=1, lineType=8)

            # if self.captureType == 'stereo':
            #     #  Compute the distance and display it under the cursor
            #     point3DList = self.stereoCalibrationProcessor.get3DPoint(self.distancePixelList)
            #     for i in range(0, len(self.distancePixelList)):
            #         point2D = self.distancePixelList[i]
            #         point3D = point3DList[i][0]
            #         xA = point2D[0]
            #         yA = point2D[1]
            #         z = point3D[2] * -2
            #         self.drawText(finalPicture,
            #                       f'{z}',
            #                       x=int(xA), y=int(yA) + 30, fontScale=2, color=(255, 0, 0))

        return finalPicture

    def drawText(self, src, text, x=0, y=0, width=0, height=0, alignH=None, alignV=None, fontScale=None, color=None):

        fontScale = Calibrator.PANEL_FONTSCALE if fontScale is None else fontScale
        color = Calibrator.PANEL_TEXT_COLOR if color is None else color
        textSize = cv2.getTextSize(text,
                                   fontFace=Calibrator.PANEL_FONTFACE,
                                   fontScale=fontScale,
                                   thickness=Calibrator.PANEL_FONTTHICKNESS)
        (label_width, label_height), baseline = textSize

        if alignH == 'center':
            x = int(x + (width - label_width) / 2)
        elif alignH == 'right':
            x = int(width - label_width)

        if alignV == 'center':
            y = int(y + (height + label_height) / 2)
        elif alignV == 'bottom':
            y = int(height - label_height / 2)
        else:
            y += label_height

        cv2.putText(src,
                    text,
                    (x, y),
                    fontFace=Calibrator.PANEL_FONTFACE,
                    fontScale=fontScale,
                    thickness=Calibrator.PANEL_FONTTHICKNESS,
                    color=color)
        return y + 15

    def drawChessBoard(self, frame, corners, isAllCorners):
        return cv2.drawChessboardCorners(frame, Calibrator.BOARD_SHAPE, corners, isAllCorners)

    def renderDisparityMap(self, imageLeft, imageRight):
        if self.disparityDisplay is not None:
            finalPicture = self.disparityDisplay.renderDisparityMap(imageLeft, imageRight,
                                                                    self.disparityDisplay.displayMode)
            self.disparityDisplay.renderWindow(finalPicture)

    ################ INPUT METHODS ###############
    def clickAndPoint(self, event, x, y, flags, param):
        scaledX = x / self.displayScaleFactor
        scaledY = y / self.displayScaleFactor
        self.mousePosition = (scaledX, scaledY)
        if event == cv2.EVENT_LBUTTONUP:
            if self.isDrawingOverLay:
                if self.lastPoint is None:
                    self.lastPoint = (scaledX, scaledY)
                else:
                    self.lineList.append([self.lastPoint, (scaledX, scaledY)])
                    self.lastPoint = None

    # def disparityPoint(self, event, x, y, flags, params):
    #     self.disparityCursorPosition = (x, y)

    def listenKeyboard(self):
        key = cv2.waitKeyEx(100)
        currentCamera = None
        currentSetting = None

        if self.captureType == 'left':
            currentCamera = self.cameraLeft
            currentSetting = self.cameraLeftSetting
        elif self.captureType == 'right':
            currentCamera = self.cameraRight or self.cameraLeft
            currentSetting = self.cameraRightSetting

        if key != -1:
            if key == ord('q') or key == 27:  # q or ESC to quit
                self.closeDisparityWindow()
                self.release()
            elif key == 35 or key == 47:  # Tilde to show debug
                self.toggleDebug()
            elif key == 43 or key == 61:  # Numpad Plus key
                self.increaseResolution()
            elif key == 45:  # Numpad Minus key
                self.decreaseResolution()
            elif key == 32:  # SpaceBar key
                self.photoManual()
            elif key == ord('h'):  # h key : horizontal flip of left camera
                if currentCamera is not None:
                    currentCamera.invertHorizontal()
                    currentSetting['hFlip'] = currentCamera.flipHorizontal
                    self.saveConfiguration()
                    # Clear picture folder as old piture could mess computation
                    self.clearFolder()
            elif key == ord('v'):  # v key : vertical flip of left camera
                if currentCamera is not None:
                    currentCamera.invertVertical()
                    currentSetting['vFlip'] = currentCamera.flipVertical
                    self.saveConfiguration()
                    # Clear picture folder as old piture could mess computation
                    self.clearFolder()
            elif key == ord('x'):  # x key : drawing tools
                self.isDrawingOverLay = not self.isDrawingOverLay
                self.lineList = []
            elif key == ord('b'):  # b key : debugging flag
                self.testBascule = not self.testBascule
            elif key == ord('m'):  # m key : enable DisparityMap
                if self.isShowDisparity == False:
                    self.openDisparityWindow()
                else:
                    self.closeDisparityWindow()

            elif key == ord('c'):  # c key : save calib
                if self.captureType == "stereo":
                    self.saveCalibration()
            elif key == ord('i'):  # v key : Invert Camera
                if self.isStereo and not self.cameraLeft.isStereoCam:
                    temp = self.cameraRight
                    # Swap cameras
                    self.cameraRight = self.cameraLeft
                    self.cameraLeft = temp
                    self.cameraIndexes = [self.cameraIndexes[1], self.cameraIndexes[0]]
                    # Swap settings
                    temp = self.cameraRightSetting
                    self.cameraRightSetting = self.cameraLeftSetting
                    self.cameraLeftSetting = temp
                    self.saveConfiguration()
                    # Clear picture folder as old piture could mess computation
                    self.clearFolder()
            elif key == 13:  # ENTER key : next processor
                if self.isStereo and self.activeProcessor is not None:
                    self.lastCornerPosition = None
                    self.lastPhotoCornerPosition = None
                    self.currentImageIndex = -1
                    if self.captureType == 'left':
                        if self.rightCalibrationProcessor is not None:
                            self.activeProcessor = self.rightCalibrationProcessor
                            self.captureType = 'right'
                    elif self.captureType == 'right':
                        self.activeProcessor = self.stereoCalibrationProcessor
                        self.captureType = 'stereo'
                    elif self.captureType == 'stereo':
                        self.activeProcessor = self.leftCalibrationProcessor
                        self.captureType = 'left'
                        self.closeDisparityWindow()
                    self.currentCalibrationIndex = self.activeProcessor.getCalibrationIndex()
                self.lineList = []
                self.closeDisparityWindow()
            elif key == 123 or key == ord('a'):  # LEFT arrow key : previous calibration results
                if self.currentCalibrationIndex is not None:
                    if self.currentCalibrationIndex > 0:
                        self.currentCalibrationIndex -= 1
            elif key == 123 or key == ord('d'):  # RIGHT arrow key : next calibration results
                if self.currentCalibrationIndex is not None:
                    calibrationCount = 0
                    if self.activeProcessor is not None:
                        calibrationCount = self.activeProcessor.getCalibrationIndex()
                    if self.currentCalibrationIndex < calibrationCount:
                        self.currentCalibrationIndex += 1
            elif key == 126 or key == ord('w'):  # UP arrow key : previous image
                if self.currentImageIndex > -1:
                    self.currentImageIndex -= 1
                self.lineList = []
            elif key == 125 or key == ord('s'):  # DOWN arrow key : next image
                imageCount = 0
                if self.activeProcessor is not None:
                    imageCount = self.activeProcessor.getImageIndex()
                if self.currentImageIndex < imageCount - 1:
                    self.currentImageIndex += 1
                self.lineList = []
            elif key == 117 or key == ord('r'):  # r  key or DELETE: remove/re-add the current picture
                if self.activeProcessor is not None:
                    self.activeProcessor.disableImage(self.currentImageIndex)
            elif key == ord('p'):  # p key preview clibration result
                self.previewCalibration = not self.previewCalibration
            elif key == ord('t'):  # t  : enabled / disable autoshoot/rotate
                self.autoShoot = not self.autoShoot
            else:
                print(f'Key pressed: {key}')

    def toggleDebug(self):
        if self.cameraLeft is not None:
            self.cameraLeft.toggleDebug()
        if self.cameraRight is not None:
            self.cameraRight.toggleDebug()

    def increaseResolution(self):
        self.displayScaleFactor += 0.1

    def decreaseResolution(self):
        self.displayScaleFactor -= 0.1
        if self.displayScaleFactor < 0.1:
            self.displayScaleFactor = 0.1

    @staticmethod
    def getChessBoardForFrame(frame, boardShape=None, useFast=True):
        # Graysquale to pass to findChessboardCorners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_FAST_CHECK if useFast else 0
        boardShape = Calibrator.BOARD_SHAPE if boardShape is None else boardShape

        isAllCorners, corners = cv2.findChessboardCorners(gray, boardShape, flags=flags)
        return isAllCorners, corners, gray

    def detectBoardShape(self, frame):
        boardShape = (self.boardRowShape, self.boardColShape)
        isAllCorners, _, _ = Calibrator.getChessBoardForFrame(frame, boardShape, useFast=True)
        if isAllCorners:
            print(f'Dimension du Damier trouvée : {boardShape}')
            Calibrator.BOARD_SHAPE = boardShape
            self.saveConfiguration()
            return boardShape
        else:
            minCol, minRow = Calibrator.BOARD_MIN_CORNER
            if self.boardColShape > minCol:
                # Try with on less row
                self.boardColShape -= 1
            elif self.boardRowShape > minRow:
                self.boardRowShape -= 1
                self.boardColShape, _ = Calibrator.BOARD_MAX_CORNER
            else:
                # Reset the detection
                self.boardColShape, self.boardRowShape = Calibrator.BOARD_MAX_CORNER
        return None

    def isInsideBoundingBox(self, center, point, halfSize):
        if center is None or point is None:
            return False

        center_x, center_y = center
        point_x, point_y = point
        if (point_x > center_x - halfSize and point_x < center_x + halfSize):
            if (point_y > center_y - halfSize and point_y < center_y + halfSize):
                return True
        return False

    def launchProcessor(self):
        if self.activeProcessor is None:
            # Launch each calibration processor
            calibrationThread = Thread(target=self.calibrationThreadRunner, args=([self.leftCalibrationProcessor]))
            calibrationThread.start()
            self.activeProcessor = self.leftCalibrationProcessor
            self.captureType = "left"
            if self.isStereo:
                calibrationThread = Thread(target=self.calibrationThreadRunner, args=([self.rightCalibrationProcessor]))
                calibrationThread.start()
                calibrationThread = Thread(target=self.calibrationThreadRunner,
                                           args=([self.stereoCalibrationProcessor]))
                calibrationThread.start()
            else:
                # No stereo calibration
                self.rightCalibrationProcessor = None
                self.stereoCalibrationProcessor = None

    def calibrationThreadRunner(self, cameraProcessor):

        # Create the grid
        rows = Calibrator.BOARD_SHAPE[0]
        cols = Calibrator.BOARD_SHAPE[1]
        chessBoardGrid = np.zeros((rows * cols, 3), np.float32)
        chessBoardGrid[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
        # Apply squareSize to the grid ( and convert mm to meter)
        chessBoardGrid *= self.squareSize / 1000.0

        # Compute Calibration
        while self.cameraLeft is not None and self.cameraLeft.isRunning():
            if self.captureType == cameraProcessor.captureType:
                if cameraProcessor.isReadyToCompute():
                    cameraProcessor.compute(chessBoardGrid, self.fishEye)
                    self.currentCalibrationIndex = cameraProcessor.getCalibrationIndex()

                time.sleep(0.5)
            else:
                time.sleep(3)
