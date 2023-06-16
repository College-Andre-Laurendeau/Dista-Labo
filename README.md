
# Dista
###### **D**etection **I**ntelligente par **S**téréoscopique et **T**raitement **A**lgorithmique 

Dista est un projet de distanciation physique par la reconnaissance d'images en temps réel.



## Calibration d'une nouvelle caméra
Aller lire le _README.md_ dans le repertoire **calibration** pour les explications


## Exemples de Commandes 
* Commande à lancer pour la caméra corridor CEGEP : 

`
DISPLAY=:0 python3 detection.py --usb --zed --sn=23336181 --resolution=HD --db=mysql:192.168.1.30/dista --hold_for=2 --vflip=1 --view_2dbox --arrows --detection_threshold=0.5 --fullscreen
`





-----------
Dista est un porté par une équipe de recehrcehr du Cegep André Laurendeau

