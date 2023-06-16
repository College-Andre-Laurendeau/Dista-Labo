################################################################################
##          Classe ChessboardImage
##      Stock les informations d'une image et les resultats de sa calibration
##  Author : J. Coupez
##  Date : 20/01/2021
##  Copyright Dista
##  Version : 1.0
################################################################################

import cv2


class ChessboardImage():

    def __init__(self, fileName):
        self.fileName = fileName
        self.image = None
        self.imageSize = None
        self.isEnabled = True
        self.isAllCorners = False
        self.rmsError = None
        self.realCorners = []
        self.distortedCorners = []

    def getIsEnabled(self):
        return self.isEnabled

    def getFileName(self):
        return self.fileName

    def getShortFileName(self):
        return self.fileName[self.fileName.rfind('/')+1:]

    def getImage(self):
        if self.image is None:
            self.image = cv2.imread(self.fileName)
            #print(f'read image {self.fileName}.') #***here
            
        return self.image

    def getImageSize(self):
        image = self.getImage()
        if image is not None:
            height, width, _ = image.shape
            return (width, height)
        return (0, 0)

    def toggleEnabled(self):
        self.isEnabled = not self.isEnabled
        self.resetCalibration()

    def disable(self):
        self.isEnabled = False
        self.resetCalibration()

    def resetCalibration(self):
        self.isAllCorners = False
        self.realCorners = []
        self.distortedCorners = []
        self.rmsError = None

    def getIsAllCorners(self):
        return self.isAllCorners

    def setIsAllCorners(self, isAllCorners):
        self.isAllCorners = isAllCorners

    def getRealCorners(self):
        return self.realCorners

    def setRealCorners(self, corners):
        self.realCorners = corners

    def getDistortedCorners(self):
        return self.distortedCorners

    def setDistortedCorners(self, corners):
        self.distortedCorners = corners

    def getRmsError(self):
        return self.rmsError

    def getRmsErrorAsString(self):
        if self.rmsError is not None:
            '''Truncates/pads a float f to n decimal places without rounding'''
            n = 5
            s = '{}'.format(self.rmsError)
            if 'e' in s or 'E' in s:
                return '{0:.{1}f}'.format(self.rmsError, n)
            i, p, d = s.partition('.')
            value = '.'.join([i, (d + '0' * n)[:n]])

            return f'RMS: {value}'
        else:
            return 'RMS: -'

    def setRmsError(self, error):
        self.rmsError = error

