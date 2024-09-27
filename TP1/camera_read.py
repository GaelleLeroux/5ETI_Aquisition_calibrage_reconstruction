import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import os
from datetime import datetime
import cmath
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

##############################
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # Fixe la hauteur du flux acquis à,→ 240 pixels
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480) # Fixe la largeur du flux acquis à,→ 480 pixels
# cap.set(cv2.CAP_PROP_FPS, 20) # Fixe la fréquence du flux à 20 images/s
# cap.set(cv2.CAP_PROP_SHARPNESS, 3) # Fixe la netteté du flux à 3
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,1) # 1 : off - 3 : on. Active,→ l'exposition automatique.
# cap.set(cv2.CAP_PROP_BRIGHTNESS,100) # Gère l'exposition du flux
# cap.set(cv2.CAP_PROP_SATURATION,100) # Gère la saturation du flux
# cap.set(cv2.CAP_PROP_CONTRAST,100) # Gère le contraste du flux
##############################

tx = 10
ty = 15
theta = np.pi
s = 0.5
alpha = s*np.cos(theta)
beta = s*np.sin(theta)

contour = False
move = False
A=100
rx=200
ry=200
height = 480
width = 640
cx = width // 2
cy = height // 2
deltax = np.zeros((height, width), dtype=np.float32)
deltay = np.zeros((height, width), dtype=np.float32)
Dx = np.zeros((height, width), dtype=np.float32)
Dy = np.zeros((height, width), dtype=np.float32)
for y in range(height):
    for x in range(width):
        z = complex((x-cx), (y-cy))
        alpha = cmath.phase(z)
        deltax[y, x] = A * np.cos(alpha) * math.exp(-1 * ((x - cx) ** 2 / rx + (y - cy) ** 2 / ry))
        deltay[y, x] = A * np.sin(alpha) * math.exp(-1 * ((x - cx) ** 2 / rx + (y - cy) ** 2 / ry))

        Dx[y, x] = x + deltax[y, x]
        Dy[y, x] = y + deltay[y, x]

while(True):
    ret, frame = cap.read() #1 frame acquise à chaque iteration
    if contour:
        edge = cv2.Canny(frame, 50, 100)
        Rho = 1  # résolution du paramètre ρ en pixels, pour avoir la fenetre avec les in de la meme taille que notre fenetre
        Theta = np.pi/180
        Threshold = 50  # nombre minimum d'intersections
        minLineLength = 100  # longueur minimum des lignes
        maxLineGap = 5  # distance maximum entre deux segments de ligne
        lines = cv2.HoughLinesP(edge,Rho,Theta,Threshold,minLineLength=200,maxLineGap=5)
        print(lines)
        res=frame
        if not(lines is None): # si lines contient au moins une ligne
            print('oui')
            for line in lines: # on itère sur les lignes détectées
                x1,y1,x2,y2 = line[0] # on récupère les coordonnées de la ligne,→ courante
                res = cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2) # on superpose la,→ ligne courante à la frame courante pour l'affichage
        cv2.imshow('Hough', res) #affichage
        cv2.imshow('Edge', edge) #affichage
                
        # else :
        #     print('non')

    if move :
        try :
            height, width, _ = frame.shape
        except :
            height, width= frame.shape
        cx = width // 2
        cy = height // 2
        M = np.float32([[alpha, beta, (1-alpha)*cx-beta*cy+tx], [-beta, alpha, beta*cx+(1-alpha)*cy+ty]])
        frame = cv2.warpAffine(frame, M, (width, height))

    # aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    # aruco_parameters = cv2.aruco.DetectorParameters()
    # aruco_detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_parameters)   
    # corners, ids, rejectedPoints = aruco_detector.detectMarkers(frame)
    # I_aruco = cv2.aruco.drawDetectedMarkers(frame,corners, ids)
    
    

    dst = cv2.remap(frame, Dx, Dy, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Transfo non lineaire', dst) #affichage



    
    
    cv2.imshow('Capture_Video', frame) #affichage
    key = cv2.waitKey(1) #on évalue la touche pressée

    if key & 0xFF == ord('o'): #si appui sur 'q'
        orb = cv2.ORB_create(nfeatures=50,nlevels=8)
        kp = orb.detect(frame, None) # Détection des "keypoints" sur la frame,→ courante
        res = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=0) #
        cv2.imshow('ORB', res) #affichage

    if key & 0xFF == ord('q'): #si appui sur 'q'
        break #sortie de la boucle while
    if key & 0xFF == ord('s'): #si appui sur 's' pour screenshot
        current_date_time = datetime.now()
        current_hour = current_date_time.hour
        folder_path = os.path.join(os.getcwd(),"screenshot")
        os.makedirs(folder_path, exist_ok=True)
        current_date_time_str = f'screenshot_{current_date_time}_{current_hour}'.replace(' ', '_').replace('.','-').replace(':','-')+'.jpg'
        screenshot_path=os.path.join(folder_path,current_date_time_str)
        plt.imsave(screenshot_path, cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

    if key & 0xFF == ord('c'): #pour afficher que les contours
        if contour:
            contour = False
        else:
            contour=True

    if key & 0xFF == ord('m'): #pour move l'image selon question 1.5
        if move:
            move = False
        else:
            move=True

cap.release()
cv2.destroyAllWindows()