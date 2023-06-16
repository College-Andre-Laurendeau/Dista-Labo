################################################################################
##          Classe CalibratorApplication
##      Gere la fenetre de configuration des caméras
##  Author : J. Coupez
##  Date : 16/06/2021
##  Copyright Dista
##  Version : 1.0
################################################################################

# TODO : Corriger la supression d'image sur la nano ( supprime pas)
# TODO : Ajouter le scale pour afficher l'image rectifiee

# SYSTEM LIBRARIES
import os
import glob
import json
from threading import Thread

# VISION + PROCESSING LIBRARIES
import cv2
import numpy as np

# GRAPHICAL LIBRARIES
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# DISTA LIBRARIES
from lib.NetCam import NetCam
from lib.ConfigurationDialog import ConfigurationDialog
from lib.CalibrationMonoProcessor import CalibrationMonoProcessor
from lib.CalibrationStereoProcessor import CalibrationStereoProcessor
from lib.CountDown import CountDown
from lib.DisparityProcessor import DisparityProcessor
from lib.DisparityDisplay import DISPARITY_DISPLAYMODE
from lib.ChessboardImage import ChessboardImage as fileImage #***here


class CALIBRATION_MODE:
    NONE = ""
    CHESSBOARD_DETECT = "Détection"
    LEFT_CAM = "Caméra Gauche"
    RIGHT_CAM = "Caméra Droite"
    STEREO = "Stéréoscopie"
    DISPARITY = "Disparité"


class CalibratorApplication():
    DEFAULT_WINDOW_NAME = 'Stéréo Calibrator, Dista @ Cegep André Laurendeau / Creative Commons'
    BOARD_MAX_CORNER = (12, 10)
    BOARD_MIN_CORNER = (6, 6)
    BOARD_SHAPE = None

    def __init__(self, args, location):
        self.args = args
        self.location = location
        self.window = Tk()

        self.configDialog = None
        self.currentConfiguration = {}
        self.refreshInterval = 100

        self.leftCalibrationProcessor = None
        self.rightCalibrationProcessor = None
        self.stereoCalibrationProcessor = None
        self.activeProcessor = None

        # Display properties
        self.serial = "TEST"
        self.calibrationMode = StringVar(value=CALIBRATION_MODE.CHESSBOARD_DETECT)
        self.cameraCanvas = None
        self.canvasWidth = 640
        self.canvasHeight = 480
        self.displayImage = None
        self.stepCanvas = None
        self.stepDrawingList = {}
        self.previous_shape = None
        self.btn_previousStep = None
        self.btn_nextStep = None
        self.list_filebrowser = None
        self.detectionDynamicLabel = StringVar()
        self.imageList = StringVar()
        self.reprojectionLabel = StringVar()
        self.disparityDownScale = IntVar(value=2)
        self.displayMode = IntVar(value=DISPARITY_DISPLAYMODE.COLOR)
        self.calibrationFilename = StringVar()
        self.blendingFactor = DoubleVar()

        # Photo Shoot
        self.autoShootEnabled = BooleanVar(value=True)
        self.photoCountDown = CountDown(5)
        self.leftCorners = []
        self.isAllLeftCorners = False
        self.rightCorners = []
        self.isAllRightCorners = False
        self.lastCornerPosition = None
        self.lastPhotoCornerPosition = None
        self.imageLeft = None
        self.imageRight = None
        self.countdownCanvas = None
        self.btn_manualShoot = None
        self.currentImageIndex = -1
        self.mousePosition = None

        # Camera Properties
        self.resolution = "VGA"
        self.isCsiCam = False
        self.squareSize = 36.2
        self.fishEye = False
        self.cameraIndexes = ["/dev/video0"]
        self.cameraLeftSetting = {}
        self.cameraRightSetting = {}
        self.cameraLeft = None
        self.cameraRight = None

        # Compute Properties
        self.disparityProcessor = None
        self.depthMap = None

        # BoardDetection Properties
        self.boardColShape, self.boardRowShape = CalibratorApplication.BOARD_MAX_CORNER

        self.createWindow()

        self.showConfigurationDialog()

    def start(self):
        self.window.mainloop()

    def createWindow(self):
        self.window.geometry("870x600+100+100")
        self.window.resizable(True, True)
        self.window.minsize(870, 600)
        self.window.title(CalibratorApplication.DEFAULT_WINDOW_NAME)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)
        self.mainframe = ttk.Panedwindow(self.window, orient=VERTICAL)
        self.mainframe.grid(column=0, row=0, sticky=N + S + E + W)

        # Top PANEL
        mainPanel = ttk.Panedwindow(self.mainframe, orient=HORIZONTAL)
        self.mainframe.add(mainPanel, weight=1)

        # LEFT PANEL
        s = ttk.Style()
        s.configure('Danger.TFrame', background='red')
        leftFrame = ttk.Frame(mainPanel)
        # leftFrame = ttk.Frame(mainPanel, style='Danger.TFrame')
        self.createLeftFrameContent(leftFrame)
        mainPanel.add(leftFrame)

        # RIGHT PANEL
        s = ttk.Style()
        s.configure('Black.TFrame', background='black')
        rightFrame = ttk.Frame(mainPanel, style='Black.TFrame')
        mainPanel.add(rightFrame, weight=1)
        self.createRightFrameContent(rightFrame)

    def createLeftFrameContent(self, frame):
        frame["padding"] = (5, 5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        lbl_configStep = ttk.Label(frame, textvariable=self.calibrationMode, anchor="center", width=30)
        lbl_configStep.grid(column=0, row=0, sticky=(N, W, E))

        # Chessboard Panel
        self.panel_chessBoard = ttk.Frame(frame)
        self.createChessBoardPanel(self.panel_chessBoard)

        # Calibration Panel
        self.panel_calibration = ttk.Frame(frame)
        self.createCalibrationPanel(self.panel_calibration)

        # Disparity Panel
        self.panel_disparity = ttk.Frame(frame)
        self.createDisparityPanel(self.panel_disparity)

        pass

    def createChessBoardPanel(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        lbl_detecting = ttk.Label(frame, text="Detection du damier en cours...", anchor="center", width=30)
        lbl_detecting.grid(column=0, row=0, sticky=(W, E))

        lbl_size = ttk.Label(frame, textvariable=self.detectionDynamicLabel, anchor="center")
        lbl_size.grid(column=0, row=1, sticky=(W, E))

    def createCalibrationPanel(self, frame):
        frame.columnconfigure(0, weight=1)

        highlightStyle = ttk.Style()
        highlightStyle.configure('Highlight.TFrame', background='#007AFF')

        panel_fileBrowser = ttk.Labelframe(frame, text="Images :")
        panel_fileBrowser.grid(column=0, row=0, sticky=(N, W, E))
        panel_fileBrowser.columnconfigure(0, weight=1)
        panel_fileBrowser.rowconfigure(0, weight=1)

        self.list_filebrowser = Listbox(panel_fileBrowser, listvariable=self.imageList)
        self.list_filebrowser.grid(column=0, row=0, sticky=(N, W, S, E))
        s = ttk.Scrollbar(panel_fileBrowser, orient=VERTICAL, command=self.list_filebrowser.yview)
        self.list_filebrowser.configure(yscrollcommand=s.set)
        self.list_filebrowser['yscrollcommand'] = s.set
        self.list_filebrowser.bind("<<ListboxSelect>>",
                                   lambda e: self.handleOnSelectImage(self.list_filebrowser.curselection()))
        s.grid(column=1, row=0, sticky=(N, S, E))

        btn_deleteFile = ttk.Button(panel_fileBrowser, text='Supprimer', command=self.handleOnDeleteFile)
        btn_deleteFile.grid(column=0, row=1, sticky=(W, E))

        # btn_disableFile = ttk.Button(panel_fileBrowser, text='Desactiver', command=self.handleOnDeleteFile)
        # btn_disableFile.grid(column=1, row=1, sticky=(W, E))

        # BACK FROM PREVIEW PANEL
        self.panel_backToPreview = ttk.Frame(frame, style='Highlight.TFrame')
        self.panel_backToPreview.columnconfigure(0, weight=1)
        self.panel_backToPreview.rowconfigure(0, weight=1)
        self.panel_backToPreview["padding"] = (10, 10)
        btn_backToPreview = ttk.Button(self.panel_backToPreview, text='Retour à la capture',
                                       command=self.handleOnDeselectImage)
        btn_backToPreview["padding"] = (15, 20)
        btn_backToPreview.grid(column=0, row=0, sticky=(W, E))

        # BLENDING PICTURE PANEL
        self.panel_blending = ttk.Labelframe(frame, text="Affichage de l'image rectifiée :")
        self.panel_blending["padding"] = (10, 10)
        scale_blending = ttk.Scale(self.panel_blending, orient=HORIZONTAL, variable=self.blendingFactor, length=200, from_=0.0, to=1.0)
        scale_blending.set(0)
        scale_blending.grid(column=0, row=1, sticky=(W, E))

        # AUTOSHOOT PANEL
        self.panel_autoshoot = ttk.Frame(frame, style='Highlight.TFrame')
        self.panel_autoshoot.grid(column=0, row=3, sticky=(W, E))
        self.panel_autoshoot.columnconfigure(0, weight=1)
        autoshoot = ttk.Checkbutton(self.panel_autoshoot, text='Prise de photo Automatique',
                                    variable=self.autoShootEnabled, command=self.handleOnAutoShootChanged,
                                    onvalue=True, offvalue=False)
        autoshoot["padding"] = (5, 5)
        autoshoot.grid(column=0, row=0, sticky=(W, E))
        self.btn_manualShoot = ttk.Button(self.panel_autoshoot, text='Prendre une photo',
                                          command=self.handleOnPhotoManual)
        self.btn_manualShoot["padding"] = (5, 20)

        # COUNTDOWN
        self.countdownCanvas = Canvas(frame, highlightthickness=0, height=151)
        diameter = 150
        start_x = 45
        start_y = 0
        self.countdownCanvas.create_oval(start_x, start_y, start_x + diameter,
                                         start_y + diameter, fill='white',
                                         outline='black')
        self.countdownText = self.countdownCanvas.create_text(start_x + diameter / 2,
                                                              start_y + diameter / 2, text="-",
                                                              font=('TkMenuFont', 80), fill='black')

        # PROGRESS COMPUTE PANEL
        self.panelComputing = ttk.Frame(frame)
        self.panelComputing.columnconfigure(0, weight=1)
        self.panelComputing.rowconfigure(0, weight=1)
        lbl_configStep = ttk.Label(self.panelComputing, text="Calcul en cours...", anchor="center")
        lbl_configStep.grid(column=0, row=0, sticky=(N, W, E))
        self.progressBarCompute = ttk.Progressbar(self.panelComputing, orient=HORIZONTAL, mode='determinate',
                                                  length=200)
        self.progressBarCompute.grid(column=0, row=1)

        # CHARTS PANEL
        self.panel_charts = ttk.Frame(frame)
        self.panel_charts.columnconfigure(0, weight=1)

        plt.ion()
        self.chartFigure = plt.figure(figsize=(1, 2), dpi=100)
        chartcanvas = FigureCanvas(self.chartFigure, self.panel_charts)
        self.rmsChart = chartcanvas.get_tk_widget()
        # self.rmsChart.grid(column=0, row=0, sticky=(N, W, E))
        # self.ax = self.chartFigure.gca()
        # self.renderCharts()

        lbl_rmsTitle = ttk.Label(self.panel_charts, text="Erreur de reprojection totale :", anchor="center", width=30)
        lbl_rmsTitle.grid(column=0, row=1, sticky=(S, W, E))
        lbl_rmsTitle["padding"] = (0, 10)
        lbl_rms = ttk.Label(self.panel_charts, textvariable=self.reprojectionLabel, anchor="center", width=30)
        lbl_rms.grid(column=0, row=2, sticky=(S, W, E))

    def createDisparityPanel(self, frame):
        frame.columnconfigure(0, weight=1)

        panel_disparity = ttk.Labelframe(frame, text="Paramétrage")
        panel_disparity.grid(column=0, row=0, sticky=(W, E))
        panel_disparity.columnconfigure(0, weight=1)
        panel_disparity["padding"] = (10, 10)

        gray = ttk.Radiobutton(panel_disparity, text='Niveau de gris', variable=self.displayMode,
                               value=DISPARITY_DISPLAYMODE.GRAY)
        gray.grid(column=0, row=5, sticky=(W, E))
        color = ttk.Radiobutton(panel_disparity, text='Dégradé de couleur', variable=self.displayMode,
                                value=DISPARITY_DISPLAYMODE.COLOR)
        color.grid(column=0, row=6, sticky=(W, E))
        image = ttk.Radiobutton(panel_disparity, text='Image', variable=self.displayMode,
                                value=DISPARITY_DISPLAYMODE.IMAGE)
        image.grid(column=0, row=7, sticky=(W, E))

        lbl_title = ttk.Label(panel_disparity, text="Division de la résolution", anchor="center", width=30)
        lbl_title.grid(column=0, row=10, sticky=(W, E))
        disparityScale = ttk.Scale(panel_disparity, orient=HORIZONTAL, length=100, from_=1, to=5,
                                   variable=self.disparityDownScale)
        disparityScale.grid(column=0, row=20, sticky=(W, E))
        lbl_title = ttk.Label(panel_disparity, textvariable=self.disparityDownScale, anchor="center", width=30)
        lbl_title.grid(column=0, row=30, sticky=(W, E))

        btn_saveDisparity = ttk.Button(frame, text='Sauvegarder la calibration', command=self.handleOnSaveCalibration)
        btn_saveDisparity.grid(column=0, row=10, sticky=(W, E), pady=20)

        self.panel_saved = ttk.Frame(frame)
        self.panel_saved.columnconfigure(0, weight=1)
        lbl_saved = ttk.Label(self.panel_saved, text='Calibration sauvegardée dans :', anchor="center")
        lbl_saved.grid(column=0, row=0, sticky=(W, E))
        lbl_filePath = ttk.Label(self.panel_saved, textvariable=self.calibrationFilename, anchor="center", width=30)
        lbl_filePath.grid(column=0, row=1, sticky=(W, E))

    def createRightFrameContent(self, frame):
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Top PANEL
        topFrame = ttk.Frame(frame)
        self.createTopFrameContent(topFrame)
        topFrame.grid(column=0, row=0, sticky=(E, W, S))

        # Center Canvas
        self.cameraCanvas = Canvas(frame, background="#000000", highlightthickness=0, width=self.canvasWidth,
                                   height=self.canvasHeight)
        self.cameraCanvas.grid(column=0, row=1)
        self.cameraCanvas.bind("<Button-1>", self.saveMousePos)
        frame.bind("<Configure>", self.onResizeCanvas)

        pass

    def createTopFrameContent(self, frame):
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        self.btn_previousStep = ttk.Button(frame, text='< Précédent', command=self.handleOnPreviousMode)
        self.btn_previousStep.grid(column=0, row=0, sticky=(N, W, S))

        self.stepCanvas = Canvas(frame, width=440, height=50)
        self.stepCanvas.grid(column=1, row=0, sticky=(N, S))
        diameter = 35
        decal_x = 30
        start_x = decal_x
        start_y = 1
        spacing = 90

        # STEP 1
        self.stepDrawingList["step_1_shape"] = self.stepCanvas.create_oval(start_x, start_y, start_x + diameter,
                                                                           start_y + diameter, fill='white',
                                                                           outline='black')
        self.stepDrawingList["step_1_text"] = self.stepCanvas.create_text(start_x + diameter / 2,
                                                                          start_y + diameter / 2, text="0",
                                                                          font='TkMenuFont', fill='black')
        self.stepDrawingList["step_1_bottom"] = self.stepCanvas.create_text(start_x + diameter / 2, diameter + 10,
                                                                            text=CALIBRATION_MODE.CHESSBOARD_DETECT,
                                                                            anchor='center', font=('TkMenuFont', 8),
                                                                            fill='black')

        # STEP 2
        start_x = start_x + spacing
        self.stepDrawingList["step_2_shape"] = self.stepCanvas.create_oval(start_x, start_y, start_x + diameter,
                                                                           start_y + diameter, fill='white',
                                                                           outline='black')
        self.stepDrawingList["step_2_text"] = self.stepCanvas.create_text(start_x + diameter / 2,
                                                                          start_y + diameter / 2, text="1",
                                                                          font='TkMenuFont', fill='black')
        self.stepCanvas.create_text(start_x + diameter / 2, diameter + 10, text=CALIBRATION_MODE.LEFT_CAM,
                                    anchor='center', font=('TkMenuFont', 8), fill='black')

        # STEP 3
        start_x = start_x + spacing
        self.stepDrawingList["step_3_shape"] = self.stepCanvas.create_oval(start_x, start_y, start_x + diameter,
                                                                           start_y + diameter, fill='white',
                                                                           outline='black')
        self.stepDrawingList["step_3_text"] = self.stepCanvas.create_text(start_x + diameter / 2,
                                                                          start_y + diameter / 2, text="2",
                                                                          font='TkMenuFont', fill='black')
        self.stepCanvas.create_text(start_x + diameter / 2, diameter + 10, text=CALIBRATION_MODE.RIGHT_CAM,
                                    anchor='center', font=('TkMenuFont', 8), fill='black')

        # STEP 4
        start_x = start_x + spacing
        self.stepDrawingList["step_4_shape"] = self.stepCanvas.create_oval(start_x, start_y, start_x + diameter,
                                                                           start_y + diameter, fill='white',
                                                                           outline='black')
        self.stepDrawingList["step_4_text"] = self.stepCanvas.create_text(start_x + diameter / 2,
                                                                          start_y + diameter / 2, text="3",
                                                                          font='TkMenuFont', fill='black')
        self.stepCanvas.create_text(start_x + diameter / 2, diameter + 10, text=CALIBRATION_MODE.STEREO,
                                    anchor='center', font=('TkMenuFont', 8), fill='black')

        # STEP 5
        start_x = start_x + spacing
        self.stepDrawingList["step_5_shape"] = self.stepCanvas.create_oval(start_x, start_y, start_x + diameter,
                                                                           start_y + diameter, fill='white',
                                                                           outline='black')
        self.stepDrawingList["step_5_text"] = self.stepCanvas.create_text(start_x + diameter / 2,
                                                                          start_y + diameter / 2, text="4",
                                                                          font='TkMenuFont', fill='black')
        self.stepCanvas.create_text(start_x + diameter / 2, diameter + 10, text=CALIBRATION_MODE.DISPARITY,
                                    anchor='center', font=('TkMenuFont', 8), fill='black')

        self.btn_nextStep = ttk.Button(frame, text='Suivant >', command=self.handleOnNextMode)
        self.btn_nextStep.grid(column=2, row=0, sticky=(N, S, E))

    def onResizeCanvas(self, configuration):
        width, height = configuration.width, configuration.height

        ratioW = 4.0
        ratioH = 3.0
        if self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
            ratioW = 8.0

        self.mousePosition = None

        self.canvasWidth = width
        self.canvasHeight = int(width / ratioW * ratioH)
        if self.canvasHeight > height:
            self.canvasHeight = height
            self.canvasWidth = int(height / ratioH * ratioW)

        self.cameraCanvas["width"] = self.canvasWidth
        self.cameraCanvas["height"] = self.canvasHeight

        # print(f'Resize Canvas to : {self.canvasWidth}x{self.canvasHeight}')

    def showConfigurationDialog(self):
        self.window.withdraw()
        self.configDialog = ConfigurationDialog(self.args, self.window, self.onCloseConfigurationDialog, self.location)

    def onCloseConfigurationDialog(self, currentConfiguration={}, clearPreviousPicture=False):
        self.configDialog = None
        self.window.update()
        self.window.deiconify()
        self.loadConfiguration(currentConfiguration)
        self.startCalibration(clearPreviousPicture)
        pass

    def loadConfiguration(self, settings):
        self.currentConfiguration = settings
        self.cameraIndexes = settings.get('cameraIndexes') or []
        self.fishEye = settings.get('fishEye') or False
        self.squareSize = settings.get('squareSize') or 1
        self.serial = settings.get("serial") or 'TEST'
        self.resolution = settings.get('resolution') or 'VGA'
        self.isCsiCam = settings.get('isCsiCam') or False
        self.cameraLeftSetting = settings.get('leftCamera') or {}
        self.cameraRightSetting = settings.get('rightCamera') or {}

    def saveConfiguration(self):
        # Save the calibration for the next run
        #  Update the boardShape

        self.currentConfiguration["boardShape"] = CalibratorApplication.BOARD_SHAPE
        jsonConfigurationFile = os.path.join(self.location,f'calibration/{self.serial}.json')
        print("self.location",self.location)
        print("saveConfiguration",jsonConfigurationFile)

        with open(jsonConfigurationFile, 'w') as outfile:
            json.dump(self.currentConfiguration, outfile, indent=4)

    def startCalibration(self, clearPreviousPicture=False):
        boardShape = self.currentConfiguration.get("boardShape") or None
        self.calibrationMode.set(CALIBRATION_MODE.CHESSBOARD_DETECT)
        if boardShape is not None and boardShape != '':
            CalibratorApplication.BOARD_SHAPE = (boardShape[0], boardShape[1])
            self.boardRowShape, self.boardColShape = CalibratorApplication.BOARD_SHAPE
            self.handleOnNextMode()

        filePath = os.path.join(self.location,f'calibration/{self.serial}')
        self.leftCalibrationProcessor = CalibrationMonoProcessor('left', filePath, self.resolution,
                                                                 CalibratorApplication.getChessBoardForFrame,
                                                                 self.onComputeProgress)
        self.rightCalibrationProcessor = CalibrationMonoProcessor('right', filePath, self.resolution,
                                                                  CalibratorApplication.getChessBoardForFrame,
                                                                  self.onComputeProgress)
        self.stereoCalibrationProcessor = CalibrationStereoProcessor('stereo', filePath, self.resolution,
                                                                     CalibratorApplication.getChessBoardForFrame,
                                                                     leftCalibrationProcessor=self.leftCalibrationProcessor,
                                                                     rightCalibrationProcessor=self.rightCalibrationProcessor,
                                                                     onComputeProgress=self.onComputeProgress)
        self.leftCalibrationProcessor.resetImageIndex()
        self.rightCalibrationProcessor.resetImageIndex()
        self.stereoCalibrationProcessor.resetImageIndex()

        self.createFolder(filePath)
        if clearPreviousPicture:
            self.clearFolder(filePath)

        self.initCapture()
        self.refreshMode()
        self.mainLoop()

    def saveCalibration(self):
        jsonConfigurationFile = None
        calibration = self.stereoCalibrationProcessor.getCoefficient()
        if calibration:
            self.currentConfiguration[self.resolution] = calibration.toJSON()
            jsonConfigurationFile = os.path.join(self.location,f'calibration/{self.serial}.json')
            self.calibrationFilename.set(jsonConfigurationFile)
            with open(jsonConfigurationFile, 'w') as outfile:
                json.dump(self.currentConfiguration, outfile, indent=4)
                print(f'Saved config to {jsonConfigurationFile}')
            with open(os.path.join(self.location,f'zedinfo/{self.serial}.conf'), 'a') as outfile:
                outfile.write(calibration.toTxt())
                outfile.close()
                print(f'Saved config to zedinfo/{self.serial}.conf')
        return jsonConfigurationFile

    ######### DISPLAY METHODS #################
    def refreshMode(self):
        id_shape = None
        id_text = None

        if self.calibrationMode.get() == CALIBRATION_MODE.CHESSBOARD_DETECT:
            self.btn_previousStep.grid_forget()
            self.btn_nextStep.grid_forget()
            self.panel_chessBoard.grid(column=0, row=1, sticky=(W, E))
        else:
            self.btn_previousStep.grid(column=0, row=0, sticky=(N, W, S))
            self.btn_nextStep.grid(column=2, row=0, sticky=(N, S, E))
            self.panel_chessBoard.grid_forget()

        if self.calibrationMode.get() == CALIBRATION_MODE.CHESSBOARD_DETECT:
            id_shape = self.stepDrawingList.get("step_1_shape")
            id_text = self.stepDrawingList.get("step_1_text")
        elif self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM:
            id_shape = self.stepDrawingList.get("step_2_shape")
            id_text = self.stepDrawingList.get("step_2_text")
        elif self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM:
            id_shape = self.stepDrawingList.get("step_3_shape")
            id_text = self.stepDrawingList.get("step_3_text")
        elif self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
            id_shape = self.stepDrawingList.get("step_4_shape")
            id_text = self.stepDrawingList.get("step_4_text")
        elif self.calibrationMode.get() == CALIBRATION_MODE.DISPARITY:
            id_shape = self.stepDrawingList.get("step_5_shape")
            id_text = self.stepDrawingList.get("step_5_text")

        if id_shape is not None:
            if self.previous_shape:
                self.stepCanvas.itemconfigure(self.previous_shape, fill='white')
                self.stepCanvas.itemconfigure(self.previous_text, fill='black')
            self.previous_shape = id_shape
            self.previous_text = id_text

            self.stepCanvas.itemconfigure(id_shape, fill='#007AFF')
            self.stepCanvas.itemconfigure(id_text, fill='white')
        self.refreshFileList()
        self.renderCharts()
        pass

    def refreshFileList(self):
        print("refreshFileList")
        if self.activeProcessor is not None:
            imageList = self.activeProcessor.chessboardImageList
            fileNameList = []
            rms = ""
            for i in range(len(imageList)):
                captureImage = imageList[i]
                if type(captureImage) is tuple:
                    fileName = captureImage[0].getShortFileName()
                    rms = captureImage[0].getRmsErrorAsString()
                else:
                    fileName = captureImage.getShortFileName()
                    rms = captureImage.getRmsErrorAsString()
                fileNameList.append(f'{fileName} ({rms})')
            self.imageList.set(fileNameList)
            self.list_filebrowser.see(len(fileNameList))

            if self.activeProcessor.isReadyToCompute():
                self.rmsChart.grid_forget()
                calibrationThread = Thread(target=self.calibrationThreadRunner, args=([self.activeProcessor]))
                calibrationThread.start()


    def mainLoop(self):
        displayImage = None
        currentCamera = None

        if self.activeProcessor and self.currentImageIndex >= 0:
            displayImage = self.activeProcessor.getImage(self.currentImageIndex)
            alpha = self.blendingFactor.get()
            if alpha > 0.0:
                #  We blend the undistorted image with the normal image
                undistortedImage = self.activeProcessor.undistort(displayImage)
                beta = (1.0 - alpha)
                displayImage = cv2.addWeighted(undistortedImage, alpha, displayImage , beta, 0.0)

        else:
            calibrationMode = self.calibrationMode.get()
            if calibrationMode == CALIBRATION_MODE.CHESSBOARD_DETECT:
                # CHESSBOARD DETECTION : TRY TO FIND THE CHESSBOARD SIZE
                displayImage = self.mainLoopChessBoard()
                currentCamera = self.cameraLeft
            elif calibrationMode == CALIBRATION_MODE.LEFT_CAM:
                #  LEFT CAMERA CALIBRATION
                displayImage = self.mainLoopLeftCamera()
                currentCamera = self.cameraLeft
                if self.autoShootEnabled.get():
                    displayImage = self.photoAutoShooter(displayImage, None, self.imageLeft, None)
            elif calibrationMode == CALIBRATION_MODE.RIGHT_CAM:
                # RIGHT CAMERA CALIBRATION
                displayImage = self.mainLoopRightCamera()
                currentCamera = self.cameraRight
                if self.autoShootEnabled.get():
                    displayImage = self.photoAutoShooter(None, displayImage, None, self.imageRight)
            elif calibrationMode == CALIBRATION_MODE.STEREO:
                # STEREO CALIBRATION ( LEFT + RIGHT)
                displayLeft = self.mainLoopLeftCamera()
                displayRight = self.mainLoopRightCamera()
                if displayLeft is not None and displayRight is not None:
                    if self.autoShootEnabled.get():
                        displayImage = self.photoAutoShooter(displayLeft, displayRight, self.imageLeft, self.imageRight)
                    else:
                        displayImage = np.concatenate((displayLeft, displayRight), axis=1)
                currentCamera = self.cameraLeft
            elif calibrationMode == CALIBRATION_MODE.DISPARITY:
                displayImage = self.mainLoopDisparity()

        if displayImage is not None:
            # Resize and change Color palette
            displayImage = cv2.cvtColor(displayImage, cv2.COLOR_BGR2RGB)
            displayImage = cv2.resize(displayImage, (self.canvasWidth, self.canvasHeight),
                                      interpolation=cv2.INTER_LINEAR)
            self.displayImage = ImageTk.PhotoImage(Image.fromarray(displayImage))
            self.cameraCanvas.create_image(1, 1, anchor=NW, image=self.displayImage)

        # DRI - Dynamic refresh interval
        refreshInterval = self.refreshInterval
        if currentCamera and currentCamera.captureFps:
            fps = currentCamera.captureFps.fps
            if fps > 0:
                refreshInterval = int(1000 / fps)
        self.window.after(refreshInterval, self.mainLoop)

    def mainLoopChessBoard(self):
        rawImage = None
        if self.cameraLeft is not None:
            rawImage = self.cameraLeft.readLeft()
            if rawImage is not None:
                self.detectBoardShape(rawImage)
        return rawImage

    def mainLoopLeftCamera(self):
        displayImage = None
        if self.cameraLeft is not None:
            self.imageLeft = self.cameraLeft.readLeft()
            if self.imageLeft is not None:
                displayImage = self.imageLeft.copy()
                self.isAllLeftCorners, self.leftCorners, _ = CalibratorApplication.getChessBoardForFrame(self.imageLeft)
                displayImage = self.drawChessBoard(displayImage, self.leftCorners, self.isAllLeftCorners)

        return displayImage

    def mainLoopRightCamera(self):
        displayImage = None
        if self.cameraRight is not None:
            if len(self.cameraIndexes) > 1:
                self.imageRight = self.cameraRight.readLeft()
            else:
                self.imageRight = self.cameraRight.readRight()
            if self.imageRight is not None:
                displayImage = self.imageRight.copy()
                self.isAllRightCorners, self.rightCorners, _ = CalibratorApplication.getChessBoardForFrame(
                    self.imageRight)
                displayImage = self.drawChessBoard(displayImage, self.rightCorners, self.isAllRightCorners)

        return displayImage

    def mainLoopDisparity(self):
        displayImage = None

        if self.disparityProcessor and self.cameraLeft and self.cameraRight:
            imageLeft = self.cameraLeft.readLeft()
            imageRight = self.cameraRight.readRight()

            rectifiedImageLeft, rectifiedImageRight = self.disparityProcessor.rectifyLeftRight(imageLeft, imageRight)
            self.disparityProcessor.disparityDownScale = int(self.disparityDownScale.get())
            cloud, disparityMap, self.depthMap = self.disparityProcessor.computeDisparityMap(rectifiedImageLeft,
                                                                                             rectifiedImageRight)
            mode = self.displayMode.get()
            if mode == DISPARITY_DISPLAYMODE.IMAGE or disparityMap is None:
                displayImage = imageLeft
            elif mode == DISPARITY_DISPLAYMODE.GRAY:
                displayImage = self.disparityProcessor.convertDisparityToGray(disparityMap)
            elif mode == DISPARITY_DISPLAYMODE.COLOR:
                displayImage = self.disparityProcessor.convertDisparityToColor(disparityMap)

            if self.mousePosition:
                width = displayImage.shape[1]
                height = displayImage.shape[0]

                x, y = self.mousePosition
                xFloat = x * (width / self.canvasWidth)
                yFloat = y * (height / self.canvasHeight)

                x = int(xFloat)
                y = int(yFloat)
                depth = round(self.disparityProcessor.getPointDepth(self.depthMap, (xFloat,yFloat)), 2)
                # Horizontal line
                cv2.line(displayImage, (x - 10, y), (x + 10, y), (255, 255, 255), thickness=1, lineType=8)
                # Vertical line
                cv2.line(displayImage, (x, y - 10), (x, y + 10), (255, 255, 255), thickness=1, lineType=8)
                cv2.putText(displayImage, f'{depth:.2f}m.', (x, y + 30),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=1.5,
                            thickness=2,
                            color=(255, 255, 255))
        return displayImage

    def renderCharts(self):
        # height, width, _ = centerPanel.shape
        if self.activeProcessor is None or len(self.activeProcessor.cameraCoeficientList) == 0:
            self.panel_charts.grid_forget()
            self.chartFigure.clear()
            self.rmsChart.grid_forget()
            print('En attente de calibration...')
            return
        else:
            self.panel_charts.grid(column=0, row=4, sticky=(W, E, S))
            self.rmsChart.grid(column=0, row=0, sticky=(N, W, E))


        # # Draw Per view RMS on the left
        # fig = Figure(figsize=(6,5), dpi=100)
        # # fig.set_figheight(height / 100.0)
        # # fig.set_figwidth(width / 200.0)
        # canvas = FigureCanvas(fig, self.panel_charts)
        # canvas.get_tk_widget().grid(column=0, row=0, sticky=(N, W , E))

        x = []
        y = []
        rms = 0

        self.chartFigure.clear()
        # self.chartFigure.canvas.draw()
        self.chartFigure.canvas.flush_events()
        self.ax = self.chartFigure.add_subplot(111)

        cameraCoeficient = self.activeProcessor.getCoefficient()
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

        self.ax.plot(x, y, label="RMS de l'image")  # Plot more data on the axes...
        self.ax.fill_between(x, rms, y)
        self.ax.set_title(f'Erreur par image (RMS)')  # Add a title to the axes.
        if self.currentImageIndex >= 0:
            self.ax.axvline(self.currentImageIndex + 1, 0, 1, label='Image actuelle', c='g', linewidth=2.0)
        self.ax.set_xlabel('Images')  # Add an x-label to the axes.
        self.ax.set_ylabel('RMS')  # Add a y-label to the axes.
        self.ax.legend()  # Add a legend.
        self.ax.grid(True)  # lines

        # combine floor division // and regular division /:
        self.reprojectionLabel.set(f'{rms // 0.00001 / 100000}')



        # self.panel_charts.grid_forget()
        # self.panel_charts.grid(column=0, row=0, sticky=(N, W, E))

    ########## PICTURE GRABBING METHODS ################
    def photoAutoShooter(self, imageLeft, imageRight, rawLeft, rawRight):

        countDownFrame = None
        currentCornerPosition = None
        savePicture = False
        calibrationMode = self.calibrationMode.get()
        if calibrationMode == CALIBRATION_MODE.LEFT_CAM:
            countDownFrame = imageLeft
            if self.isAllLeftCorners:
                currentCornerPosition = self.leftCorners[0][0]
        elif calibrationMode == CALIBRATION_MODE.RIGHT_CAM:
            countDownFrame = imageRight
            if self.isAllRightCorners:
                currentCornerPosition = self.rightCorners[0][0]
        elif calibrationMode == CALIBRATION_MODE.STEREO:
            #countDownFrame = imageLeft
            #countDownFrame = imageRight here***
            countDownFrame = np.concatenate((imageLeft, imageRight), axis=1)
            if self.isAllLeftCorners and self.isAllRightCorners:
                currentCornerPosition = self.leftCorners[0][0]

        if self.lastCornerPosition is None:
            self.lastCornerPosition = currentCornerPosition

        if currentCornerPosition is not None:
            if self.isInsideBoundingBox(self.lastCornerPosition, currentCornerPosition, 10):
                # the first corner is near the currentCorner
                if not self.isInsideBoundingBox(self.lastPhotoCornerPosition, currentCornerPosition, 20):
                    if self.photoCountDown.isFinished:
                        # We take only 1 picture in the same area (corner must have moved by a square of 40 pixels)
                        savePicture = True
                        self.lastPhotoCornerPosition = currentCornerPosition
                        self.hideCountdown()
                    elif not self.photoCountDown.isRunning:
                        self.photoCountDown.start()


            else:
                # The corner has moved since last time : Reinit the countdown
                self.photoCountDown.stop()
                self.hideCountdown()
                self.lastCornerPosition = None
                self.lastPhotoCornerPosition = None
        else:
            # No Corner found : stop the countdown
            self.photoCountDown.stop()
            self.hideCountdown()

        if countDownFrame is not None:
            if self.photoCountDown.isRunning:
                ### Display countDown
                remainingTime = self.photoCountDown.getRemainingSeconds()
                if (remainingTime < 4):
                    self.showCountdown()
                    self.countdownCanvas.itemconfigure(self.countdownText, text=f'{remainingTime}')
            elif self.lastPhotoCornerPosition is not None:
                ### Display photoshoot
                pt1 = (0, 0)
                height, width, _ = countDownFrame.shape
                pt2 = (width, height)
                overlay = countDownFrame.copy()
                overlay = cv2.rectangle(overlay, pt1, pt2, (255, 255, 255), -1)
                countDownFrame = cv2.addWeighted(overlay, 0.4, countDownFrame, 1 - 0.4, 0, countDownFrame)
                countDownFrame = cv2.putText(countDownFrame,
                                             f'Image prise ! deplacez le damier...',
                                             (int(width / 2 - 220), int(height / 2 + 40)),
                                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        if savePicture:
            if calibrationMode == CALIBRATION_MODE.LEFT_CAM:
                self.leftCalibrationProcessor.savePicture(rawLeft)
            elif calibrationMode == CALIBRATION_MODE.RIGHT_CAM:
                self.rightCalibrationProcessor.savePicture(rawRight)
            elif calibrationMode == CALIBRATION_MODE.STEREO:
                self.stereoCalibrationProcessor.savePicture(rawLeft, rawRight)
            self.refreshFileList()

        finalPicture = countDownFrame

        return finalPicture

    def isInsideBoundingBox(self, center, point, halfSize):
        if center is None or point is None:
            return False

        center_x, center_y = center
        point_x, point_y = point
        if (point_x > center_x - halfSize and point_x < center_x + halfSize):
            if (point_y > center_y - halfSize and point_y < center_y + halfSize):
                return True
        return False

    def showCountdown(self):
        self.countdownCanvas.grid(column=0, row=3, sticky=(E, W))

    def hideCountdown(self):
        self.countdownCanvas.grid_forget()


    ######### EVENT HANDLERS METHODS #################
    def saveMousePos(self, event):
        mouse_x, mouse_y = event.x, event.y
        self.mousePosition = (mouse_x, mouse_y)

    def handleOnNextMode(self):
        self.handleOnDeselectImage()
        if self.calibrationMode.get() == CALIBRATION_MODE.CHESSBOARD_DETECT:
            self.stepCanvas.itemconfigure(self.stepDrawingList.get("step_1_shape"), fill='#5abd1c')
            self.stepCanvas.itemconfigure(self.stepDrawingList.get("step_1_text"), text="✔", fill='white',
                                          font=('TkMenuFont', 22))
            self.stepCanvas.itemconfigure(self.stepDrawingList.get("step_1_bottom"),
                                          text=f'Damier : ({self.boardRowShape} x {self.boardColShape})')
            self.panel_chessBoard.grid_forget()
            self.previous_shape = None
            # next step : LEFT CAM
            self.calibrationMode.set(CALIBRATION_MODE.LEFT_CAM)
            self.panel_calibration.grid(column=0, row=1, sticky=(N, W, S, E))
        elif self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM:
            self.calibrationMode.set(CALIBRATION_MODE.RIGHT_CAM)
        elif self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM:
            self.calibrationMode.set(CALIBRATION_MODE.STEREO)
        elif self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
            self.calibrationMode.set(CALIBRATION_MODE.DISPARITY)
            self.panel_calibration.grid_forget()
            self.panel_disparity.grid(column=0, row=1, sticky=(N, W, S, E))
            self.handleOnDeselectImage()
            self.initDisparityProcessor()
        elif self.calibrationMode.get() == CALIBRATION_MODE.DISPARITY:
            self.calibrationMode.set(CALIBRATION_MODE.LEFT_CAM)
            self.panel_disparity.grid_forget()
            self.panel_saved.grid_forget()
            self.panel_calibration.grid(column=0, row=1, sticky=(N, W, S, E))

        self.initCapture()
        self.refreshMode()

    def handleOnPreviousMode(self):
        self.handleOnDeselectImage()
        if self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM:
            self.calibrationMode.set(CALIBRATION_MODE.DISPARITY)
            self.panel_calibration.grid_forget()
            self.panel_disparity.grid(column=0, row=1, sticky=(N, W, S, E))
            self.handleOnDeselectImage()
            self.initDisparityProcessor()
        elif self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM:
            self.calibrationMode.set(CALIBRATION_MODE.LEFT_CAM)
        elif self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
            self.calibrationMode.set(CALIBRATION_MODE.RIGHT_CAM)
        elif self.calibrationMode.get() == CALIBRATION_MODE.DISPARITY:
            self.calibrationMode.set(CALIBRATION_MODE.STEREO)
            self.panel_disparity.grid_forget()
            self.panel_saved.grid_forget()
            self.panel_calibration.grid(column=0, row=1, sticky=(N, W, S, E))

        self.initCapture()
        self.refreshMode()

    def handleOnAutoShootChanged(self):
        if self.autoShootEnabled.get():
            self.btn_manualShoot.grid_forget()
        else:
            self.btn_manualShoot.grid(column=0, row=2, sticky=(W, E), pady=10, padx=10)
            self.hideCountdown()

    def handleOnPhotoManual(self):
        calibrationMode = self.calibrationMode.get()
        if calibrationMode == CALIBRATION_MODE.LEFT_CAM:
            self.leftCalibrationProcessor.savePicture(self.imageLeft)
        elif calibrationMode == CALIBRATION_MODE.RIGHT_CAM:
            self.rightCalibrationProcessor.savePicture(self.imageRight)
        elif calibrationMode == CALIBRATION_MODE.STEREO:
            self.stereoCalibrationProcessor.savePicture(self.imageLeft, self.imageRight)
        self.refreshFileList()

    def handleOnDeleteFile(self): #here***

        if self.activeProcessor:
            chessboardImage = self.activeProcessor.getChessboardImage(self.currentImageIndex)           
            if chessboardImage:
                imageIndex = self.currentImageIndex
                imageSerial = self.serial	      
                if self.calibrationMode.get() == CALIBRATION_MODE.STEREO:      
                    print("On est au stereo")
                    #print(f"serial : {imageSerial}")
                    #print(f"index image : {imageIndex}")             
                    try:
                        for root, dirs, files in os.walk(f"/home/dista/Documents/dista/calibration/{imageSerial}/stereo"):
                            files.sort()
                            fileLeftSelect = files[imageIndex]   
                            fileRightSelect = files[imageIndex + 1]
                            fileLeftDelete = os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/{fileLeftSelect}")
                            fileRightDelete = os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/{fileRightSelect}")
                            print(f"delete left image : {fileLeftSelect}")
                            print(f"delete right image : {fileRightSelect}")
                            
                        """if imageIndex in range(0,10): 
                            os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/00{imageIndex}_left.jpg")      
                            os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/00{imageIndex}_right.jpg")
                        elif imageIndex in range(10,100):
                            os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/0{imageIndex}_left.jpg")      
                            os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/0{imageIndex}_right.jpg")
                        elif imageIndex >= 100:
                            os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/{imageIndex}_left.jpg")      
                            os.remove(f"/home/dista/Documents/dista/calibration/{self.serial}/stereo/{imageIndex}_right.jpg")"""
                        #os.remove(filename)
                        #if self.calibrationMode.get() == CALIBRATION_MODE.STEREO 
                        #os.remove(rightDel)
                    except OSError as e:
                        print("Error: %s : %s" % (filename, e.strerror)) 		     		                                
                elif self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM or self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM:
                    print("Droite ou gauche")
                    print(f"index image : {self.currentImageIndex}")
                    filename = chessboardImage.getFileName()
                    #print(f"son nom est {filename}")
                    try:
                        os.remove(filename)
                        #if self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
                        #os.remove(rightDel)
                    except OSError as e:
                        print("Error: %s : %s" % (filename, e.strerror))
            self.activeProcessor.resetImageIndex()
            self.refreshFileList()

    def handleOnSelectImage(self, currentSelection):
        try:
            self.currentImageIndex = currentSelection[0]
            print(f"indexSelection = {currentSelection[0]}")
            self.panel_backToPreview.grid(column=0, row=1, sticky=(W, E))
            if self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM or self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM:
                self.panel_blending.grid(column=0, row=2, sticky=(W, E),pady=20)
            self.panel_autoshoot.grid_forget()
            self.renderCharts()

        except IndexError as e:
            pass

    def handleOnDeselectImage(self):
        self.rmsChart.grid_forget()
        self.list_filebrowser.selection_clear(self.currentImageIndex)
        self.currentImageIndex = -1
        self.panel_backToPreview.grid_forget()
        self.panel_blending.grid_forget()
        self.panel_autoshoot.grid(column=0, row=3, sticky=(W, E))
        self.renderCharts()

    def handleOnSaveCalibration(self):
        self.saveCalibration()
        self.panel_saved.grid(column=0, row=20, sticky=(W, E))

    ######### CHESSBOARD DETECTION METHODS #################
    @staticmethod
    def getChessBoardForFrame(frame, boardShape=None, useFast=True):
        # Graysquale to pass to findChessboardCorners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flags = cv2.CALIB_CB_FAST_CHECK if useFast else 0
        boardShape = CalibratorApplication.BOARD_SHAPE if boardShape is None else boardShape

        isAllCorners, corners = cv2.findChessboardCorners(gray, boardShape, flags=flags)
        return isAllCorners, corners, gray

    def detectBoardShape(self, frame):
        boardShape = (self.boardRowShape, self.boardColShape)
        isAllCorners, _, _ = CalibratorApplication.getChessBoardForFrame(frame, boardShape, useFast=True)
        if isAllCorners:
            print(f'Dimension du Damier trouvée : {boardShape}')
            CalibratorApplication.BOARD_SHAPE = boardShape
            self.saveConfiguration()
            self.handleOnNextMode()
            return boardShape
        else:
            minCol, minRow = CalibratorApplication.BOARD_MIN_CORNER
            if self.boardColShape > minCol:
                # Try with on less row
                self.boardColShape -= 1
            elif self.boardRowShape > minRow:
                self.boardRowShape -= 1
                self.boardColShape, _ = CalibratorApplication.BOARD_MAX_CORNER
            else:
                # Reset the detection
                self.boardRowShape, self.boardColShape = CalibratorApplication.BOARD_MAX_CORNER
            self.detectionDynamicLabel.set(f'Recherche : ({self.boardRowShape} x {self.boardColShape})')

        return None

    def drawChessBoard(self, frame, corners, isAllCorners):
        return cv2.drawChessboardCorners(frame, CalibratorApplication.BOARD_SHAPE, corners, isAllCorners)

    ######### COMPUTING METHODS #################
    def calibrationThreadRunner(self, cameraProcessor):
        print("Start calibration")
        self.showProgress()
        # Create the grid
        rows = CalibratorApplication.BOARD_SHAPE[0]
        cols = CalibratorApplication.BOARD_SHAPE[1]
        chessBoardGrid = np.zeros((rows * cols, 3), np.float32)
        chessBoardGrid[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
        # Apply squareSize to the grid ( and convert mm to meter)
        chessBoardGrid *= self.squareSize / 1000.0

        # Compute Calibration
        cameraProcessor.compute(chessBoardGrid, self.fishEye)

        self.hideProgress()
        self.refreshFileList()
        self.renderCharts()


    def showProgress(self):
        self.panelComputing.grid(column=0, row=5, sticky=(W, E), pady=30)
        self.panel_charts.grid_forget()
        total = self.activeProcessor.getProgressTotal()
        self.progressBarCompute.configure(maximum=total, value=0)

    def hideProgress(self):
        self.panelComputing.grid_forget()
        self.panel_charts.grid(column=0, row=4, sticky=(W, E, S))

    def onComputeProgress(self):
        self.progressBarCompute.step()
        pass

    ######### INIT & CLEANUP METHODS #################

    def initCapture(self):
        isUseLeftCamera = False
        isUseRightCamera = False

        leftSource = None
        leftSetting = None

        rightSource = None
        rightSetting = None

        if self.calibrationMode.get() == CALIBRATION_MODE.CHESSBOARD_DETECT:
            self.activeProcessor = None
            isUseLeftCamera = True
        elif self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM:
            self.activeProcessor = self.leftCalibrationProcessor
            isUseLeftCamera = True
        elif self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM:
            self.activeProcessor = self.rightCalibrationProcessor
            isUseRightCamera = True
        elif self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
            self.activeProcessor = self.stereoCalibrationProcessor
            isUseLeftCamera = True
            isUseRightCamera = True
        elif self.calibrationMode.get() == CALIBRATION_MODE.DISPARITY:
            self.activeProcessor = None
            isUseLeftCamera = True
            isUseRightCamera = True

        if isUseLeftCamera:
            # Grab the settings for left Cam
            leftSetting = self.cameraLeftSetting
            if len(self.cameraIndexes) > 0:
                leftSource = self.cameraIndexes[0]

        if isUseRightCamera:
            # Grab the settings for right Cam
            rightSetting = self.cameraRightSetting
            if len(self.cameraIndexes) > 1:
                rightSource = self.cameraIndexes[1]
            else:
                rightSource = self.cameraIndexes[0]

        if leftSource:
            if self.cameraRight and self.cameraRight.source == leftSource:
                # If the left cam is the same as the right cam source : we reuse the right camera
                self.cameraLeft = self.cameraRight
            else:
                # Create a new source
                self.cameraLeft = self.initCamera(leftSource, self.cameraLeft, leftSetting)
        else:
            if self.cameraLeft and self.cameraLeft.source != rightSource:
                # Camera Left is initialized but is not used : Need to close it
                self.releaseLeftCamera()

        if rightSource:
            if self.cameraLeft and self.cameraLeft.source == rightSource:
                self.cameraRight = self.cameraLeft
            else:
                self.cameraRight = self.initCamera(rightSource, self.cameraRight, rightSetting)
        else:
            if self.cameraRight and self.cameraRight.source != leftSource:
                # Camera Right is initialized but is not used: Need to close it
                self.releaseRightCamera()

    def initCamera(self, source, camera, cameraSetting):
        if source is not None:
            print("source",source)
                
            if camera is None or camera.source != source:
                if camera is not None:
                    camera.release()

                isStereoCam = True
                if len(self.cameraIndexes) > 1:
                    isStereoCam = False
                print("isStereoCam",isStereoCam)
            
                camera = NetCam(source=source, capture=self.resolution, isStereoCam=isStereoCam, isCsiCam=self.isCsiCam)
                print("camera initialized", source)
                # camera.showDebug()
                if cameraSetting.get('vFlip'):
                    print("invertVertical",cameraSetting.get('vFlip'))
                    camera.invertVertical()
                if cameraSetting.get('hFlip'):
                    print("invertHorizontal",cameraSetting.get('hFlip'))
                    camera.invertHorizontal()
        return camera

    def initDisparityProcessor(self):
        self.disparityProcessor = None
        self.mousePosition = None
        jsonConfigurationFile = self.saveCalibration()
        if jsonConfigurationFile is not None and self.imageLeft is not None:
            settings = {}
            settings['resolution'] = self.resolution
            settings['configFileName'] = jsonConfigurationFile
            height, width, _ = self.imageLeft.shape
            settings['imageSize'] = [width, height]
            self.disparityProcessor = DisparityProcessor(settings=settings)

    def createFolder(self, path):
        if not os.path.exists('calibration'):
            os.makedirs('calibration')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(f'{path}/left'):
            os.makedirs(f'{path}/left')
        if not os.path.exists(f'{path}/right'):
            os.makedirs(f'{path}/right')
        if not os.path.exists(f'{path}/stereo'):
            os.makedirs(f'{path}/stereo')

    def clearFolder(self, path):
        if self.calibrationMode.get() == CALIBRATION_MODE.LEFT_CAM:
            path = path + "/left/*.jpg"
            self.leftCalibrationProcessor.resetImageIndex()
        elif self.calibrationMode.get() == CALIBRATION_MODE.RIGHT_CAM:
            path = path + "/right/*.jpg"
            self.rightCalibrationProcessor.resetImageIndex()
        elif self.calibrationMode.get() == CALIBRATION_MODE.STEREO:
            path = path + "/stereo/*.jpg"
            self.stereoCalibrationProcessor.resetImageIndex()
        else:
            # Delete all tutures
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

    def release(self):
        self.releaseLeftCamera()
        self.releaseRightCamera()

        if self.disparityProcessor is not None:
            self.disparityProcessor.release()
            self.disparityProcessor = None

        self.cameraCanvas = None
        self.window = None
        self.mainframe = None
        if self.configDialog is not None:
            self.configDialog.release()
            self.configDialog = None

    def releaseLeftCamera(self):
        if self.cameraLeft is not None:
            self.cameraLeft.release()
            self.cameraLeft = None

    def releaseRightCamera(self):
        if self.cameraRight is not None:
            self.cameraRight.release()
            self.cameraRight = None
