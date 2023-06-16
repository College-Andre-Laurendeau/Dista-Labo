################################################################################
##          Calibration de camera
##      Fichier principal pour lancer une calibration de caméra avec stereo (ou sans)
##  Cet utilitaire recupere la liste des caméras disponibles et propose des options au travers d'un menu
##  Author : J. Coupez
##  Date : 10/05/2021
##  Version : 1.0
################################################################################

import os
from config import *
from lib.CalibratorApplication import CalibratorApplication

args = parse_args()

if __name__ == '__main__':
    mainWindow = None
    try:
        location = os.path.realpath(os.path.join(os.getcwd(),os.path.dirname(__file__)))
        print("launching calibration in ", location)
        mainWindow = CalibratorApplication(args,location)
        mainWindow.start()
    except RuntimeError as ex:
        print(ex)

    finally:
        if mainWindow is not None:
            mainWindow.release()
