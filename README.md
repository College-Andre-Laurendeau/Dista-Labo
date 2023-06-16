<html lang="fr-ca">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width" />
		Protocole d'utilisation Dista, Par GRIPCAL
	</head>
	<body>
		<p></p>
		<h1>1. Introduction <img align="right" src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image046.png?raw=true" alt="Logo dista" /></h1>
		<p>L’intelligence artificielle prend de plus en plus d’importance technologiquement et cette tendance va sans doute se maintenir. Nous serons donc tous appelés, un jour ou l’autre, à travailler sur des projets exploitant l’intelligence artificielle. Il est donc essentiel d’en connaître les grandes lignes afin de pouvoir assister et communiquer avec son équipe. C’est dans cette optique que vous participez à ce laboratoire sur l’intelligence artificielle qui servira d’introduction</p>
		<h2>1.1 Voici le matériel à votre disposition :</h2>
		<ul>
			<li>Jetson Nano 4gb de Nvidia</li>
			<li>Caméra jumelle stéréoscopique IMX219-83</li>
			<li>Moniteur</li>
			<li>Câble HDMI</li>
			<li>Clavier</li>
			<li>Souris</li>
			<li>Ruban à mesurer</li>
			<li>Damier de calibration 9x6 avec taille de case de 36,2 mm</li>
			<li>Objets divers (voiture, figurines, humains, ustensiles, clavier, souris, livre, etc.)</li>
		</ul>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image001.png?raw=true" alt="image des différents objet à avoir pour le laboratoire" />
		<p><div style="font-size:large"><B>Figure 1 — Matériel du laboratoire</B></div></p>
		<p>Voici un lien vers le site waveshare pour la nano, les caméras ainsi que le boitier : </p>
		<ul>
			<li><a href="https://www.waveshare.com/product/ai/accessories/cases/jetson-nano-case-c.htm">Metal Case</a></li>
			<li><a href="https://www.waveshare.com/product/ai/boards-kits/jetson-nano/jetson-nano-dev-kit-a.htm?sku=21881">Jetson Nano avec Caméras</a></li>
		</ul>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/Nano_Diagram.png?raw=true" alt="Shéma de la Jetson Nano" />
		<p><div style="font-size:large"><B>Figure 2 — Shéma de la Jetson Nano</B></div></p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/Nano_Description.PNG?raw=true" alt="Description du shéma de la jetson Nano" />
		<p><div style="font-size:large"><B>Figure 3 — Description du shéma</B></div></p>
		<h2>2.    Calibration de la caméra stéréoscopique</h2>
		<p>2.1 Lancez l’application de calibration.</p>
		<p>2.2 Lorsque le panneau de configuration de l’application de calibration s’affiche (voir figure 3), sélectionnez les paramètres de configuration suivants :</p>
		<ul>
			<li>Nouvelle configuration</li>
			<li>Nom de configuration : SN"UnNuméroDeVotreChoix" (le numéro est important alors assurez-vous de le noter)</li>
			<li>Résolution : NANO</li>
			<li>Taille d’une case : 36,2 mm</li>
			<li>Caméra CSI</li>
			<li>Taille du damier : 9 x 6</li>
			<li>Caméra gauche : video0 (1er choix du menu déroulant)</li>
			<li>Caméra droite : vidéo1 (2e choix du menu déroulant)</li>
			<li>Inversion horizontale et verticale</li>
		</ul>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image004.png?raw=true" alt="Exemple de la première page de calibration" />
		<p><div style="font-size:large"><B>Figure 4 — Panneau de configuration de l’application de calibration</B></div></p>
		<p>2.3 Vérifiez que l’image de gauche vient de la caméra de gauche.</p>
		<p>2.4 Calibration de la lentille gauche (Étape 1 de l’application)</p>
		<p>Cette étape consiste à prendre 6 « images » différentes du damier de calibration.</p>
		<p>Voici les étapes à suivre :</p>
		<p>2.4.1       Positionnez le damier de calibration à une distance de la caméra de sorte que l’image affichée soit complètement remplie par ce dernier.</p>
			<p>2.4.2       Lorsque le système aura détecté le damier, vous verrez des lignes de couleurs apparaitre sur l’image du damier.</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image006.png?raw=true" alt="exemple de détection du damier" />
		<p><div style="font-size:large"><B>Figure 5 — Damier correctement détecté</B></div></p>
		<p>2.4.3       Stabilisez le damier jusqu’à ce que le décompte de 3 à 1 s’affiche à l’écran suivi du message « Image prise ! Déplacez le damier ».</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image008.png?raw=true" alt="exemple de photo correctement prise" />
		<p><div style="font-size:large"><B>Figure 6 — Image prise avec succès</B></div></p>
		<p>2.4.4       Prenez une 2e image à la même distance (en respectant les étapes précédentes) en appliquant cette fois-ci au damier une inclinaison d’environ 30 degrés vers le haut (toujours attendre le message « Image prise ! Déplacez le damier »).</p>
		<p>2.4.5       Prenez une 3e image à la même distance (en respectant les étapes précédentes) en appliquant cette fois-ci au damier une inclinaison d’environ 30 degrés vers le bas.</p>
		<p>2.4.6       Prenez une 4e puis une 5e image avec des inclinaisons vers la gauche et vers la droite d’environ 30 degrés.</p>
		<p>2.4.7       Positionnez maintenant le damier de calibration à environ 1 mètre de la caméra.</p>
		<p>2.4.8       Prenez une image à cette position.</p>
		<p>2.4.9       Cela vous donnera un total de 6 captures d’images avec l’objectif d’obtenir une erreur de reprojection inférieure à 0,2 (voir figure 6).</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image010.png?raw=true" alt="Exemple de compilation d'image" />
		<p><div style="font-size:large"><B>Figure 7 — L’erreur de reprojection est affichée en bas à gauche de l’écran</B></div></p>
		<h3>Exemples de captures d’images</h3>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image012.png?raw=true" alt="Exemple d'image prise" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image014.png?raw=true" alt="Exemple d'image prise" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image016.png?raw=true" alt="Exemple d'image prise" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image018.png?raw=true" alt="Exemple d'image prise" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image020.png?raw=true" alt="Exemple d'image prise" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image022.png?raw=true" alt="Exemple d'image prise" />
		<p><div style="font-size:large"><B>Figure 8 — Exemples de captures d’images</B></div></p>
		<p>2.5 Calibration de la lentille droite (Étape 2 de l’application)</p>
		<ul>
			<li>Répétez la même procédure que pour la lentille gauche.</li>
		</ul>
		<p>2.6 Calibration de la stéréo (Étape 3 de l’application)</p>
		<ul>
			<li>6 captures, en suivant la même procédure qu’aux étapes précédentes.</li>
			<li>L’objectif est d’atteindre une erreur inférieure à 0,2.</li>
		</ul>
		<h3>Exemples de captures d’images en stéréo</h3>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image024.png?raw=true" alt="Exemple d'image stéréo" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image026.png?raw=true" alt="Exemple d'image stéréo" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image028.png?raw=true" alt="Exemple d'image stéréo" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image030.png?raw=true" alt="Exemple d'image stéréo" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image032.png?raw=true" alt="Exemple d'image stéréo" />
		<p><div style="font-size:large"><B>Figure 9 — Exemples de captures d’images en stéréo</B></div></p>
		<p>2.7 Visualisation de la carte de profondeur (Étape 4 de l’application)</p>
		<p>Cette étape vous permettra de vérifier que votre calibration est adéquate pour poursuivre le laboratoire.</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image034.png?raw=true" alt="Feed caméra pour confirmer la calibration" />
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image036.png?raw=true" alt="Feed caméra pour confirmer la calivration" />
		<p><div style="font-size:large"><B>Figure 10 — Outil de visualisation de la carte de profondeur</B></div></p>
		<ul>
			<li>Alternez entre les deux types de visualisation (nuance de gris et dégradé de couleurs).</li>
			<li>Tentez de comprendre quelle couleur représente les objets les plus proches et quelle couleur les objets les plus éloignés.</li>
		</ul>
		<p>Vous venez de produire une carte de profondeur. C’est-à-dire, une image pour laquelle la profondeur de chaque pixel est déterminée. Cela procure une information qui n’était pas disponible dans ni l’une ni l’autre des deux images produites par les deux caméras, mais qui est issue d’une analyse de la différence, ou de la « disparité », entre les deux images.</p>
		<ul>
			<li> Après avoir sauvegardé votre calibration, vous pouvez quitter l’application en fermant toutes les fenêtres.</li>
		</ul>
		<h2>3.    Détection d’objets et prise de mesures</h2>
		<p>3.1 Test de détection d’objets (sans calcul de distances)</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image038.png?raw=true" alt="Exemple de détection d'objet" />
		<p><div style="font-size:large"><B>Figure 11 — Exemples de détections d’objects</B></div></p>
		<p>Le réseau de neurones artificiels utilisé pour la détection d’objets est un modèle « léger » spécialement conçu pour des machines peu puissantes comme votre Jetson Nano. Plusieurs paramètres interviennent dans la qualité des détections obtenues : qualité de l’image, luminosité, environnement visuel, seuil de détection, etc.</p>
		<p>3.1.1 Depuis votre terminal, aller dans le documents "Dista-main" qui se trouvera à l'endroit où vous l'avez télécharger ou déplacé.</p>
		<p>3.1.2 Entrer ensuite cette ligne de code "python3 detection.py --opencv --usb --resolution=NANO --catego=car,cellphone,mouse --view_2dbox --sn=<span style="color:red">votreNuméro</span> --linuxid=0,1 --yolo4  --displaydistance --vflip=1,1 --hflip=1,1 --rectifydown=2 --justdetec"</p>
		<p>N.B. Vous devez changer la partie "votreNuméro" pour le nom que vous avez donné à votre calibration sans ré-écrire le SN.</p>
		<p>L’activation de la caméra et le chargement initial en mémoire du modèle de détection sont longs. Veuillez patienter…</p>
		<p>3.1.3 Placez une voiture, une souris d’ordinateur et un téléphone cellulaire à différents endroits du champ de la caméra et à différentes distances.</p>
		<p>3.1.4 Déplacez-les pour essayer le plus de possibilités possible.</p>
		<p>Cela vous permettra de comprendre les zones dans lesquelles il sera préférable de ne pas travailler, faute de détections fiables.</p>
		<p>3.1.5 Pour fermer le programme, entrer un ctrl-c dans votre terminal. Si vous fermer la fenêtre de caméra avec le x rouge, elle ne fera que se réouvrir automatiquement immédiatement après.</p>
		<p>3.2 Mesures de distances à la caméra</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image043.png?raw=true" alt="Exemple de calcul de distance" />
		<p><div style="font-size:large"><B>Figure 12 — Affichage de la distance</B></div></p>
		<p>N.B. Pour alléger la charge de calcul, les distances sont mises à jour toutes les 3 secondes environ.</p>
		<p>3.2.1 Observer les distances calculées suite à votre calibration en copiant la commande suivante de votre terminal : "python3 detection.py --opencv --usb --resolution=NANO --catego=car,cellphone,mouse --view_2dbox --sn=<span style="color:red">votreNuméro</span> --linuxid=0,1 --yolo4  --displaydistance --vflip=1,1 --hflip=1,1 --rectifydown=2"</p>
		<p>3.2.2 Choisissez un objet facile à détecter et déposez-le devant la caméra. La distance entre l’objet et la caméra devrait s’afficher.</p>
		<p>3.2.3 Vous pouvez mesurer les distances avec un ruban à mesurer afin de comparer vos résultats, vous pourez ainsi mieux comprendre les limites de votre appareil et de votre calibration.</p>
		<p>3.2.4 Fermer le programme</p>
		<p>3.3 Mesures de distances entre objets</p>
		<img src="https://raw.githubusercontent.com/College-Andre-Laurendeau/Dista-Labo/Laboratoire-Nano/readme_res/image045.png?raw=true" alt="Exemple de calcul de distance entre objets" />
		<p><div style="font-size:large"><B>Figure 13 — Distance entre objets</B></div></p>
		<p>Les calculs de stéréoscopie génèrent les coordonnés des objets dans l’espace. Il est donc possible de calculer la distance entre les objets.</p>
		<p>3.3.1 Observez les distances calculées entre les objets en copiant la commande suivante de votre terminal : "python3 detection.py --opencv --usb --resolution=NANO --catego=car,cellphone,mouse --view_2dbox --sn=<span style="color:red">votreNuméro</span> --linuxid=0,1 --yolo4  --displaydistance --vflip=1,1 --hflip=1,1 --rectifydown=2 --arrows"</p>
		<p>3.3.2 Choisissez deux objets faciles à détecter et déposez-les devant la caméra. La distance entre les deux objets devrait s’afficher.</p>
		<p>3.3.3 Comme pour le programme précédent, vous pouvez mesurer les distances avec un ruban à mesurer afin de comparer vos résultats.</p>
		<h2>4.    Changer les objets identifiables</h2>
		<p>4.1 Pour changer les objets identifiables par la Nano, simplement enlever ou ajouter des éléments dans votre code, exemple : "python3 detection.py --opencv --usb --resolution=NANO --catego=car,cellphone,mouse,person --view_2dbox --sn=<span style="color:red">votreNuméro</span> --linuxid=0,1 --yolo4  --displaydistance --vflip=1,1 --hflip=1,1 --rectifydown=2 --arrows"</p>
		<p>Voici la liste des objets que le programme peux reconnaitre :</p>
		<ul>
			<li>car</li>
			<li>cellphone</li>
			<li>mouse</li>
			<li>person</li>
		</ul>
	</body>	

</html>