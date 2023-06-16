################################################################################
##          Classe ConfigurationDialog
##      Gere la fenetre de configuration des caméras
##  Author : J. Coupez
##  Date : 16/06/2021
##  Copyright Dista
##  Version : 1.0
################################################################################
import time

import cv2
import json
import glob
import numpy as np
import os
import subprocess

from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

from lib.NetCam import NetCam


class ConfigurationDialog():
    DEFAULT_RESOLUTION = "VGA"
    DEFAULT_SQUARESIZE = 36.2

    def __init__(self, args, rootWindow,onCloseConfigurationDialog, location):
        self.refreshInterval = 20
        self.location = location
        self.rootWindow = rootWindow
        self.onCloseConfigurationDialog = onCloseConfigurationDialog
        self.isRunning = False
        # Specific camera info
        self.nerian = args.nerian or False
        self.isCsiCam = BooleanVar(value=False)

        self.resolution = StringVar(value=ConfigurationDialog.DEFAULT_RESOLUTION)
        self.fishEye = BooleanVar(value=False)

        # ChessBoard Infos
        self.squareSize = DoubleVar(value=ConfigurationDialog.DEFAULT_SQUARESIZE)
        self.isAutoDetectBoardShape = BooleanVar(value=True)
        self.shapeX = IntVar(value=0)
        self.shapeY = IntVar(value=0)

        # Calibration file info
        self.currentConfiguration = None
        self.configurationName = StringVar()
        self.configurationName.trace("w", self.handleOnConfigurationSelect)
        self.serial = StringVar()
        self.serial.trace("w", self.handleOnNameChange)
        self.scriptPath = os.getcwd()
        self.existingCalibrationList = self.retrieveConfigFileList()

        # Camera information
        self.cameraList = self.retrieveCameras()
        self.leftCamSource = StringVar()
        self.rightCamSource = StringVar()
        self.camLeftCanvas = None
        self.camRightCanvas = None

        self.leftCamera = None
        self.leftInverseH = BooleanVar(value=False)
        self.leftInverseV = BooleanVar(value=False)
        self.leftInverseH.trace("w", self.handleOnLeftFlip)
        self.leftInverseV.trace("w", self.handleOnLeftFlip)

        self.rightCamera = None
        self.rightInverseH = BooleanVar(value=False)
        self.rightInverseV = BooleanVar(value=False)
        self.rightInverseH.trace("w", self.handleOnRightFlip)
        self.rightInverseV.trace("w", self.handleOnRightFlip)

        self.windowTitle = "Configuration"
        self.previewShape = [320, 240]
        try:
            self.createWindow()
            self.refreshWindow()
            self.refreshPicture()
            self.isRunning = True

        except Exception as err:
            print(err)

    def retrieveConfigFileList(self):
        fileNameList = np.sort(glob.glob(os.path.join(self.location,f'calibration/*.json')))
        fileNameList = [w.replace(self.location, "") for w in fileNameList.tolist()]

        return fileNameList

    def retrieveCameras(self):
        """
        Test the ports and returns a tuple with the available ports and resolution that are working.
        """
        working_ports = []
        try:
            isSearching = True
            dev_port = 0
            error = 0
            while isSearching:
                isFound = False
                try:
                    process = subprocess.Popen(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
                    out, err = process.communicate()
                    lineList = out.splitlines()
                    i = 0
                    pattern = "b'(.*?)'"

                    while i < len(lineList):
                        cameraName = lineList[i].decode("utf-8").replace(":", '')
                        i += 1
                        cameraPort = lineList[i].decode("utf-8").replace("\t", '')
                        working_ports.append((cameraName, cameraPort))
                        i += 1
                        while i < len(lineList):
                            value = lineList[i].decode("utf-8")
                            i += 1
                            if len(value) == 0:
                                break
                    isSearching = False
                except RuntimeError as ex:
                    isSearching = True
                except FileNotFoundError as ex:
                    isSearching = True
                if isSearching:
                    #  v4l2 didn't worked, try the brute force
                    try:
                        camera = cv2.VideoCapture(dev_port)
                        if camera is None or not camera.isOpened():
                            isSearching = False
                        else:
                            is_reading, img = camera.read()
                            w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                            h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                            if is_reading:
                                working_ports.append((f'Caméra {dev_port} : {w}x{h}', dev_port))
                                print(f'..', end='')
                            camera.release()
                    except RuntimeError as ex:
                        error += 1
                    finally:
                        dev_port += 1

            if self.nerian:
                import PySpin
                # FLIR camera detection
                # Retrieve singleton reference to system object
                system = PySpin.System.GetInstance()
                # Retrieve list of cameras from the system
                cam_list = system.GetCameras()
                cam = None
                try:
                    for i, cam in enumerate(cam_list):
                        # Retrieve TL device nodemap and print device information
                        nodemap = cam.GetTLDeviceNodeMap()
                        # node_device_information = PySpin.CCategoryPtr(nodemap.GetNode('DeviceInformation'))
                        cameraName = None
                        node_device_name = PySpin.CStringPtr(nodemap.GetNode('DeviceModelName'))
                        node_device_serial = PySpin.CStringPtr(nodemap.GetNode('DeviceSerialNumber'))
                        if PySpin.IsAvailable(node_device_name) and PySpin.IsReadable(node_device_name):
                            cameraName = node_device_name.GetValue()
                        if PySpin.IsAvailable(node_device_serial) and PySpin.IsReadable(node_device_serial):
                            cameraName = f'{cameraName} Serial : {node_device_serial.GetValue()}'
                        dev_port = f'flir://{i}'
                        if cameraName is not None:
                            working_ports.append((f'FLIR {cameraName}', dev_port))

                except RuntimeError as ex:
                    print('Error while detecting FLIR camera', ex)
                finally:
                    del cam
                    # Clear camera list before releasing system
                    cam_list.Clear()
                    # Release system instance
                    system.ReleaseInstance()
        except:
            print('Error while detecting  camera')

        return working_ports

    def initCameras(self, leftCamSource, rightCamSource, forceReload=False):

        if rightCamSource is None :
            rightCamSource = ""
        isStereoCam = True if rightCamSource == "" else False
        isStereoCam = False if self.isCsiCam.get() else isStereoCam

        # print(f'initCameras: {leftCamSource} - {rightCamSource} - {isStereoCam}')

        if self.leftCamera is not None:
            if self.leftCamera.isStereoCam != isStereoCam:
                # Reload the left camera if the stereo status has changed
                forceReload = True
        # print(f'reloadLeftCamera: {forceReload}')

        if self.leftCamSource.get() != leftCamSource or forceReload:
            self.leftCamSource.set(leftCamSource)
            if self.leftCamera is not None:
                self.leftCamera.release()
                self.leftCamera = None
                time.sleep(2)
            if leftCamSource != "":
                self.leftCamera = NetCam(source=leftCamSource, capture="QVGA", isStereoCam=isStereoCam,
                                     isCsiCam=self.isCsiCam.get())
                if self.leftInverseH.get():
                    self.leftCamera.invertHorizontal()
                if self.leftInverseV.get():
                    self.leftCamera.invertVertical()

        if self.rightCamSource.get() != rightCamSource or forceReload:
            self.rightCamSource.set(rightCamSource)
            if self.rightCamera is not None:
                self.rightCamera.release()
                self.rightCamera = None
                time.sleep(2)
            if rightCamSource != "":
                self.rightCamera = NetCam(source=rightCamSource, capture="QVGA", isStereoCam=False,
                                          isCsiCam=self.isCsiCam.get())
                if self.rightInverseH.get():
                    self.rightCamera.invertHorizontal()
                if self.rightInverseV.get():
                    self.rightCamera.invertVertical()

        isLeftCamEnabled = True
        isRightCamEnabled = True
        if self.leftCamera is None:
            isLeftCamEnabled = False
        if self.rightCamera is None:
            isRightCamEnabled = False

        # set_value('combo_leftCameraList', self.leftCamSource)
        # set_value('combo_rightCameraList', self.rightCamSource)
        #
        # configure_item("left_hFlip", enabled=isLeftCamEnabled)
        # configure_item("left_vFlip", enabled=isLeftCamEnabled)
        # configure_item("right_hFlip", enabled=isRightCamEnabled)
        # configure_item("right_vFlip", enabled=isRightCamEnabled)

    def createWindow(self):
        self.window = Toplevel(self.rootWindow)
        self.window.title(self.windowTitle)
        self.window.geometry("800x600+30+30")
        self.window.resizable(FALSE, FALSE)
        self.window.columnconfigure(0, weight=1)
        # self.window.rowconfigure(0, weight=1)
        # self.window.rowconfigure(1, weight=1)
        self.window.protocol("WM_DELETE_WINDOW", self.handleOnContinue)
        self.window.attributes("-topmost", 1)

        topFrame = ttk.Frame(self.window)
        self.createTopFrameContent(topFrame)

        optionsFrame = ttk.Frame(self.window)
        self.createOptionsFrameContent(optionsFrame)

        cameraFrame = ttk.Frame(self.window)
        cameraFrame["padding"] = (5, 10)
        cameraFrame.grid(sticky=(W, E))
        cameraFrame.columnconfigure(0, weight=1)
        cameraFrame.rowconfigure(0, weight=1)
        cameraFrame.columnconfigure(1, weight=1)

        cameraNameList = []
        for i in range(len(self.cameraList)):
            cameraNameList.append(self.cameraList[i][0])

        leftCameraFrame = ttk.Labelframe(cameraFrame, text="GAUCHE")
        leftCameraFrame.grid(column=0, row=0)
        self.createCameraFrameContent("left", leftCameraFrame,
                                      self.leftCamSource,
                                      cameraNameList,
                                      self.leftInverseH,
                                      self.leftInverseV)
        rightCameraFrame = ttk.Labelframe(cameraFrame, text="DROITE")
        rightCameraFrame.grid(column=1, row=0)
        self.createCameraFrameContent("right", rightCameraFrame,
                                      self.rightCamSource,
                                      cameraNameList,
                                      self.rightInverseH,
                                      self.rightInverseV)

        bottomFrame = ttk.Frame(self.window)
        self.createBottomFrameContent(bottomFrame)

    def createTopFrameContent(self, frame):
        # frame["borderwidth"]=2
        # frame["relief"]="sunken"
        frame["padding"] = (5, 10)
        frame.grid(sticky=N)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        lbl_selectConfig = ttk.Label(frame, text="Selectionnez une configuration :")
        lbl_selectConfig.grid(column=0, row=0)
        combo_calibrationList = ttk.Combobox(frame, textvariable=self.configurationName)
        itemList = self.existingCalibrationList
        itemList.insert(0, "Nouvelle configuration...")
        combo_calibrationList["values"] = itemList
        combo_calibrationList.grid(column=1, row=0)
        self.grp_serial = ttk.Frame(frame)
        self.grp_serial["padding"] = (5, 5)
        lbl_configName = ttk.Label(self.grp_serial, text="Nom de la configuration :")
        lbl_configName.grid(column=0, row=0)
        txt_configName = ttk.Entry(self.grp_serial, textvariable=self.serial)
        txt_configName.grid(column=1, row=0, sticky=W)


    def createBottomFrameContent(self, frame):
        frame["padding"] = (5, 10)
        frame.grid(sticky=S)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.btn_continue = ttk.Button(frame, text='Continuer', command=self.handleOnContinue)
        self.btn_continue.grid(column=0, row=0,sticky=(N,E,S,W))


    def createOptionsFrameContent(self, frame):
        # frame["borderwidth"]=2
        # frame["relief"]="sunken"

        frame["padding"] = (5, 10)
        frame.grid()
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        # Resolution
        lbl_selectConfig = ttk.Label(frame, text="Résolution :")
        lbl_selectConfig.grid(column=0, row=0, sticky=E)
        combo_calibrationList = ttk.Combobox(frame, textvariable=self.resolution)
        combo_calibrationList["values"] = ('QVGA', 'VGA', 'NANO', 'HD', 'FHD', '2K', '5MP', '2.2K')
        combo_calibrationList.grid(column=1, row=0, sticky=W)

        # Taille carreau
        lbl_squaresize = ttk.Label(frame, text="Taille d'une case :")
        lbl_squaresize.grid(column=0, row=1, sticky=E)
        txt_squaresize = ttk.Entry(frame, textvariable=self.squareSize)
        txt_squaresize.grid(column=1, row=1, sticky=W)
        lbl_cm = ttk.Label(frame, text="mm")
        lbl_cm.grid(column=2, row=1)

        # FishEye
        check_fisheye = ttk.Checkbutton(frame, text="Lentille Fish Eye", variable=self.fishEye, onvalue=True,
                                        offvalue=False)
        check_fisheye.grid(column=1, row=2, sticky=W)
        # is CSI CAM
        check_isCSICam = ttk.Checkbutton(frame, text="Camera connectee par CSI", variable=self.isCsiCam, onvalue=True,
                                         offvalue=False,command=self.handleOnCSIChange)
        check_isCSICam.grid(column=1, row=3, sticky=W)

        # Nombre Case
        check_autodetection = ttk.Checkbutton(frame, text="Auto-detection du damier",
                                              variable=self.isAutoDetectBoardShape,
                                              onvalue=True,
                                              offvalue=False,
                                              command=self.handleOnAutoDetectSelect)
        check_autodetection.grid(column=1, row=4, sticky=W)
        shapeFrame = ttk.Frame(frame)
        txt_shapeX = Spinbox(shapeFrame, from_=1, to=30, textvariable=self.shapeX, width=5)
        txt_shapeX.grid(column=1, row=1, sticky=E)
        lbl_squaresize = ttk.Label(shapeFrame, text="x")
        lbl_squaresize.grid(column=2, row=1)
        txt_shapeX = Spinbox(shapeFrame, textvariable=self.shapeY, from_=1, to=30, width=5)
        txt_shapeX.grid(column=3, row=1, sticky=W)

        self.boardShapeFrame = shapeFrame

    def createCameraFrameContent(self, side, frame, camSource, camList, invertH, invertV):
        # Resolution
        frame["padding"] = (5, 5)
        lbl_selectConfig = ttk.Label(frame, text="Caméra :")
        lbl_selectConfig.grid(column=0, row=0, sticky=E)
        combo_cameraList = ttk.Combobox(frame, textvariable=camSource)
        combo_cameraList["values"] = camList
        combo_cameraList.grid(column=1, row=0, sticky=(E,W))
        canvas = Canvas(frame, width=self.previewShape[0], height=self.previewShape[1], background="#000000")
        canvas.grid(column=0, row=1, columnspan=2)
        if side == "left":
            self.camLeftCanvas = canvas
            combo_cameraList.bind("<<ComboboxSelected>>",self.handleOnLeftCameraSelect)
        else:
            self.camRightCanvas = canvas
            combo_cameraList.bind("<<ComboboxSelected>>",self.handleOnRightCameraSelect)
        # Invert Horizontal
        check_invertH = ttk.Checkbutton(frame, text="Invertion Horizontale", variable=invertH, onvalue=True,
                                        offvalue=False)
        check_invertH.grid(column=0, row=2, columnspan=2)
        # Invert Vertical
        check_invertV = ttk.Checkbutton(frame, text="Invertion Verticale", variable=invertV, onvalue=True,
                                        offvalue=False)
        check_invertV.grid(column=0, row=3, columnspan=2)

    def refreshWindow(self,a=False):
        # print("refreshWindow")

        if self.isAutoDetectBoardShape.get():
            self.boardShapeFrame.grid_forget()
        else:
            self.boardShapeFrame.grid(column=1, row=5)
        if self.configurationName.get() != "Nouvelle configuration...":
            self.grp_serial.grid_forget()
        else:
            self.grp_serial.grid(column=0, row=1, columnspan=2)


        enabledContinueBtn = True
        # print("serial :", self.serial.get())
        # print("self.leftCamera :", self.leftCamera)
        if self.serial.get() == "":
            enabledContinueBtn = False
        elif self.leftCamera is None:
            enabledContinueBtn = False

        if enabledContinueBtn:
            self.btn_continue.state(['!disabled'])
        else:
            self.btn_continue.state(['disabled'])

        return

    def refreshPicture(self):
        try:
            if self.isRunning:
                self.camLeftPicture = None
                self.camRightPicture = None

                if self.leftCamera is not None:
                    self.camLeftPicture = cv2.cvtColor(
                        self.leftCamera.readLeft(),
                        cv2.COLOR_BGR2RGB)

                if self.rightCamera is not None:
                    self.camRightPicture = cv2.cvtColor(self.rightCamera.readLeft(), cv2.COLOR_BGR2RGB)
                elif self.leftCamera is not None and not self.isCsiCam.get():
                    self.camRightPicture = cv2.cvtColor(self.leftCamera.readRight(), cv2.COLOR_BGR2RGB)

                if self.camLeftPicture is not None:
                    self.camLeftPicture = Image.fromarray(self.camLeftPicture)
                    self.camLeftPicture = ImageTk.PhotoImage(self.camLeftPicture)
                    self.camLeftCanvas.create_image(1, 1, anchor=NW, image=self.camLeftPicture)

                if self.camRightPicture is not None:
                    self.camRightPicture = Image.fromarray(self.camRightPicture)
                    self.camRightPicture = ImageTk.PhotoImage(self.camRightPicture)
                    self.camRightCanvas.create_image(1, 1, anchor=NW, image=self.camRightPicture)

                #     self.camRightPicture = self.emptyPictureData



        except (OSError, RuntimeError, TypeError, NameError, ValueError) as err:
            print("refreshPicture", err)
            # Swallow the exception
            return

        self.window.after(self.refreshInterval, self.refreshPicture)

        return


    def saveConfiguration(self):
        # Save the calibration for the next run
        cameraIndexes = []
        if self.leftCamSource.get() != "":
            cameraIndexes.append(self.leftCamSource.get())
        if self.rightCamSource.get() != "":
            cameraIndexes.append(self.rightCamSource.get())
        boardShape = None
        if self.shapeX.get() > 0 and self.shapeY.get() > 0:
            boardShape = [self.shapeX.get(), self.shapeY.get()]
        jsonConfigurationFile = os.path.join(self.location,f'calibration/{self.serial.get()}.json')
        
        self.currentConfiguration = {'cameraIndexes': cameraIndexes,
                                     'serial': self.serial.get(),
                                     'fishEye': self.fishEye.get(),
                                     'squareSize': self.squareSize.get(),
                                     'resolution': self.resolution.get(),
                                     'isCsiCam': self.isCsiCam.get(),
                                     'boardShape': boardShape or '',
                                     'leftCamera': {
                                         'vFlip': self.leftInverseV.get(),
                                         'hFlip': self.leftInverseH.get(),
                                     },
                                     'rightCamera': {
                                         'vFlip': self.rightInverseV.get(),
                                         'hFlip': self.rightInverseH.get(),
                                     },
                                     }
        with open(jsonConfigurationFile, 'w') as outfile:
            json.dump(self.currentConfiguration, outfile, indent=4)

    def handleOnConfigurationSelect(self, a, b,c):
        self.currentConfiguration = None
        self.serial.set("")
        if self.configurationName.get() != "Nouvelle configuration...":
            # configure_item("grp_newconfig", show=False)
            try:
                with open(f'{self.location}/{self.configurationName.get()}') as json_file:
                    self.currentConfiguration = json.load(json_file)
                    if self.currentConfiguration is not None:
                        self.serial.set(self.currentConfiguration.get("serial") or "")
                        self.resolution.set(
                            self.currentConfiguration.get("resolution") or ConfigurationDialog.DEFAULT_RESOLUTION)
                        self.fishEye.set(self.currentConfiguration.get("fishEye") or False)
                        self.squareSize.set(
                            self.currentConfiguration.get("squareSize") or ConfigurationDialog.DEFAULT_SQUARESIZE)
                        self.isCsiCam.set(self.currentConfiguration.get("isCsiCam") or False)
                        boardShape = self.currentConfiguration.get("boardShape") or None
                        if boardShape is None:
                            self.shapeX.set(0)
                            self.shapeY.set(0)
                            self.isAutoDetectBoardShape.set(True)
                        else:
                            self.shapeX.set(boardShape[0])
                            self.shapeY.set(boardShape[1])
                            self.isAutoDetectBoardShape.set(False)

                        # Left camera configuration
                        leftCamConfig = self.currentConfiguration.get("leftCamera") or None
                        if leftCamConfig is None:
                            self.leftInverseH.set(False)
                            self.leftInverseV.set(False)
                        else:
                            self.leftInverseH.set(leftCamConfig.get("hFlip") or False)
                            self.leftInverseV.set(leftCamConfig.get("vFlip") or False)

                        # Right camera configuration
                        rightCamConfig = self.currentConfiguration.get("rightCamera") or None
                        if rightCamConfig is None:
                            self.rightInverseH.set(False)
                            self.rightInverseV.set(False)
                        else:
                            self.rightInverseH.set(rightCamConfig.get("hFlip") or False)
                            self.rightInverseV.set(rightCamConfig.get("vFlip") or False)
                        cameraIndexes = self.currentConfiguration.get("cameraIndexes") or None
                        if cameraIndexes is not None:
                            leftCamSource = None
                            rightCamSource = None
                            if len(cameraIndexes) > 0:
                                leftCamSource = cameraIndexes[0]
                            if len(cameraIndexes) > 1:
                                rightCamSource = cameraIndexes[1]
                            # print(self.currentConfiguration)
                            self.initCameras(leftCamSource, rightCamSource,forceReload=True)
            except IOError as err:
                print(err)

        self.refreshWindow()

    def handleOnNameChange(self,a,b,c):
        self.refreshWindow()


    def handleOnAutoDetectSelect(self):
        if self.isAutoDetectBoardShape.get():
            self.shapeX.set(0)
            self.shapeY.set(0)
        self.refreshWindow()
        return


    def handleOnCSIChange(self):
        self.initCameras(self.leftCamSource.get(), self.rightCamSource.get(), forceReload=True)

    def handleOnLeftCameraSelect(self, a):
        self.handleOnCameraSelect("left", self.leftCamSource.get())

    def handleOnRightCameraSelect(self, a):
        self.handleOnCameraSelect("right", self.rightCamSource.get())

    def handleOnCameraSelect(self, sender, source):
        camSource = None
        camera = None
        for i in range(len(self.cameraList)):
            if source == self.cameraList[i][0]:
                camSource = self.cameraList[i][1]
        print(f'camSource: {camSource}')

        if "left" in sender:
            if self.leftCamSource.get() == self.rightCamSource.get():
                self.initCameras(camSource, camSource)
            else:
                self.initCameras(camSource, self.rightCamSource.get())
        else:
            self.initCameras(self.leftCamSource.get(), camSource)

        self.refreshWindow()
        return

    def handleOnLeftFlip(self, a, b, c):
        if self.leftCamera is not None:
            self.leftCamera.flipHorizontal = self.leftInverseH.get()
            self.leftCamera.flipVertical = self.leftInverseV.get()
        return

    def handleOnRightFlip(self, a, b, c):
        if self.rightCamera is not None:
            self.rightCamera.flipHorizontal = self.rightInverseH.get()
            self.rightCamera.flipVertical = self.rightInverseV.get()
        return

    def handleOnContinue(self):
        self.saveConfiguration()
        self.release()
        self.onCloseConfigurationDialog(self.currentConfiguration, False)

    def release(self):
        print("release")
        self.isRunning = False

        if self.leftCamera:
            self.leftCamera.release()
            self.leftCamera = None
            time.sleep(1)
        if self.rightCamera:
            self.rightCamera.release()
            self.rightCamera = None
            time.sleep(1)

        if self.window:
            self.window.destroy()


class Spinbox(ttk.Entry):

    def __init__(self, master=None, **kw):

        ttk.Entry.__init__(self, master, "ttk::spinbox", **kw)
    def set(self, value):
        self.tk.call(self._w, "set", value)
