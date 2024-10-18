# %% [markdown]
# Imports

# %%
import numpy as np 
import cv2
from matplotlib import pyplot as plt 
import math

# %% [markdown]
# Dictionnaire

# %%
taille_a = 16.3

aruco_ppm = [[[164,0,150],[164-taille_a,0,150],[164-taille_a,0,150-taille_a],[164,0,150-taille_a]],
             [[0,248,194],[0,248+taille_a,194],[0,248+taille_a,194-taille_a],[0,248,194-taille_a]],
             [[0,274,146],[0,274+taille_a,146],[0,274+taille_a,146-taille_a],[0,274,146-taille_a]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,170,166],[0,170+taille_a,166],[0,170+taille_a,166-taille_a],[0,170,166-taille_a]],
             [[0,104,199],[0,104+taille_a,199],[0,104+taille_a,199-taille_a],[0,104,199-taille_a]],
             [[373,0,181],[356.7,0,181],[356.7,0,164.7],[373,0,164.7]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
             [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
             ]
# taille_a = 163
# aruco_ppm = [[[1640,0,1500],[1640-taille_a,0,1500],[1640-taille_a,0,1500-taille_a],[1640,0,1500-taille_a]],
#              [[0,2480,1940],[0,2480+taille_a,1940],[0,2480+taille_a,1940-taille_a],[0,2480,1940-taille_a]],
#              [[0,2740,1460],[0,2740+taille_a,1460],[0,2740+taille_a,1460-taille_a],[0,2740,1460-taille_a]],
#              [[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
#              [[0,1700,1660],[0,1700+taille_a,1660],[0,1700+taille_a,1660-taille_a],[0,1700,1660-taille_a]],
#              [[0,1040,1990],[0,1040+taille_a,1990],[0,1040+taille_a,1990-taille_a],[0,1040,1990-taille_a]],
#              [[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
#              [[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
#              [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]
#              ]



# %% [markdown]
# Lecture image

# %%
frame = cv2.imread('tags.jpg')
if frame is None:
    print("Erreur : Impossible de charger l'image.")
else:
    # Afficher l'image dans une fenêtre
    plt.imshow(frame)
    plt.axis('off')  # Ne pas afficher les axes
    plt.show()

# %% [markdown]
# Lecture aruco

# %%
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_parameters = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(dictionary=aruco_dict, detectorParams=aruco_parameters)   
corners, ids, rejectedPoints = aruco_detector.detectMarkers(frame)
I_aruco = cv2.aruco.drawDetectedMarkers(frame,corners, ids)
plt.imshow(frame)
plt.axis('off')  # Ne pas afficher les axes
plt.show()

# %% [markdown]
# MAJ dictionnaire 

# %%
bibliotheque = {
    "tag": [],
    "c_mm": [],
    "c_ppx": []
    }
for i, id in enumerate(ids.flatten()):
    print("id detecte : ",id)
    bibliotheque["tag"].append(int(id))
    bibliotheque["c_mm"].append(aruco_ppm[int(id)-1])


    corners_list = corners[i][0]  # Récupère les coins du marqueur
    
    corners_list_compute = [corners_list[0].tolist(), # Haut gauche
                            corners_list[1].tolist(), # Haut droit
                            corners_list[2].tolist(), # Bas droit
                            corners_list[3].tolist()] # Bas gauche
    # Assigner les coins dans l'ordre, sans listes imbriquées
    bibliotheque["c_ppx"].append(corners_list_compute)  
    
for cle, value in bibliotheque.items() :
    print(cle," ",value)

# %% [markdown]
# Parametres

# %%
largeur = 16.3
h,l,_ = frame.shape
i2,i1 = h/2,l/2
print("i1 : ",i1)
print("i2 : ",i2)
focal = 5.7
focal = 25/1000

# s1 = abs(bibliotheque['c_ppx'][0][0][0]-bibliotheque['c_ppx'][0][1][0])/largeur
# s2 = abs(bibliotheque['c_ppx'][0][0][1]-bibliotheque['c_ppx'][0][1][1])/largeur

N = len(ids)
U1 = np.zeros((N*4,1))
A = np.zeros((N*4,7))

nmb = 0

for i in range(N):
    for j in range(4):
        U1[nmb*4+j][0] = bibliotheque['c_ppx'][i][j][0]-i1

        A[nmb*4+j][0] = (bibliotheque['c_ppx'][i][j][1]-i2)*bibliotheque['c_mm'][i][j][0]
        A[nmb*4+j][1] = (bibliotheque['c_ppx'][i][j][1]-i2)*bibliotheque['c_mm'][i][j][1]
        A[nmb*4+j][2] = (bibliotheque['c_ppx'][i][j][1]-i2)*bibliotheque['c_mm'][i][j][2]

        A[nmb*4+j][3] = bibliotheque['c_ppx'][i][j][1]-i2

        A[nmb*4+j][4] = -(bibliotheque['c_ppx'][i][j][0]-i1)*bibliotheque['c_mm'][i][j][0]
        A[nmb*4+j][5] = -(bibliotheque['c_ppx'][i][j][0]-i1)*bibliotheque['c_mm'][i][j][1]
        A[nmb*4+j][6] = -(bibliotheque['c_ppx'][i][j][0]-i1)*bibliotheque['c_mm'][i][j][2]
    nmb+=1

print(A)
print(U1)


# %% [markdown]
# Résolution de L

# %%
A_inv = np.linalg.pinv(A)
print(A_inv.shape)
print(U1.shape)
L = A_inv @ U1

print(L)


# %% [markdown]
# Résolution parametre

# %% [markdown]
# plt.imshow(im orginal)
# plt.plot(coord pixel dim1, coord pixel dim2,'RX') cela va projeter de spoints rouge au coordonne fixé par dessus l'image

# %%
O2c = 1/math.sqrt(L[5-1]**2+L[6-1]**2+L[7-1]**2) # valeur sur l'axe 2 de l'origine du repère objet dans le repère monde

beta = O2c * math.sqrt(L[1-1]**2+L[2-1]**2+L[3-1]**2) 
O1c = L[4-1]*O2c/beta
r11 = L[1-1]*O2c/beta
r12 = L[2-1]*O2c/beta
r13 = L[3-1]*O2c/beta
r21 = L[5-1]*O2c
r22 = L[6-1]*O2c
r23 = L[7-1]*O2c


r31 = r12*r23-r13*r22
r32 = r21*r13-r23*r11
r33 = r11*r22-r12*r21

theta = -math.atan(r23/r33)
gamma = -math.atan(r12/r11)
omega = -math.atan(r13/(-r23*np.sin(theta)+r33*np.cos(theta)))
print("O2C : ",O2c)
print("O1C : ",O1c)

print('*'*50)
print("r11 : ",r11)
print("r12 : ",r12)
print("r13 : ",r13)

print('*'*50)
print("r21 : ",r21)
print("r22 : ",r22)
print("r23 : ",r23)

print('*'*50)
print("r31 : ",r31)
print("r32 : ",r32)
print("r33 : ",r33)
print('*'*50)
print("theta : ",theta)
print("gamma : ",gamma)
print("omega : ",omega)

# %% [markdown]
# Résolution de O3c et f2

# %%
B = np.zeros((N*4,2))
R = np.zeros((N*4,1))

nmb = 0
for i in range(N):
    for j in range(4):
        B[nmb*4+j][0] =  bibliotheque['c_ppx'][i][j][1]-i2
        # print(bibliotheque['c_ppx'][i][j][1]-i2)
        B[nmb*4+j][1] =  -(r21*bibliotheque['c_mm'][i][j][0]+
                     r22*bibliotheque['c_mm'][i][j][1]+
                     r23*bibliotheque['c_mm'][i][j][2]+
                     O2c)
        
        R[nmb*4+j][0] = -(bibliotheque['c_ppx'][i][j][1]-i2)*(r31*bibliotheque['c_mm'][i][j][0]+
                                                            r32*bibliotheque['c_mm'][i][j][1]+
                                                            r33*bibliotheque['c_mm'][i][j][2])
        
        print(bibliotheque['c_ppx'][i][j][1]-i2, bibliotheque['c_mm'][i][j][0], bibliotheque['c_mm'][i][j][1], bibliotheque['c_mm'][i][j][2])
    nmb+=1

        
[O3c,f2] = np.linalg.pinv(B) @ R
print("O3c : ",O3c)
print("f2 : ",f2)

f1 = beta*f2
s1 = focal/f1
s2 = focal/f2

print("f1 : ",f1)
print("s1 : ",s1)
print("s2 : ",s2)

# %% [markdown]
# Projection ensemble coordonnées monde sur image acquise

# %%
Mint = np.zeros((3,4))
Mint[0][0] = focal/s1
Mint[0][2] = i1
Mint[1][1] = focal/s2
Mint[1][2] = i2
Mint[2][2] = 1
print("Mint : ",Mint)

Mext = np.zeros((4,4))
Mext[0][0] = r11
Mext[0][1] = r12
Mext[0][2] = r13
Mext[0][3] = O1c

Mext[1][0] = r21
Mext[1][1] = r22
Mext[1][2] = r23
Mext[1][3] = O2c

Mext[2][0] = r31
Mext[2][1] = r32
Mext[2][2] = r33
Mext[2][3] = O3c

Mext[3][3] = 1

M = Mint @ Mext
print("M : ",M)


# %%
u0 = []
u1 = []
for i in range(N):
    for j in range(4):
        xO1 = bibliotheque['c_mm'][i][j][0] 
        xO2 = bibliotheque['c_mm'][i][j][1] 
        xO3 = bibliotheque['c_mm'][i][j][2] 
        alpha = r31*xO1 + r32*xO2 + r33*xO3 + O3c

        Uf = M @ [xO1,xO2,xO3,1]
        
        u0.append(Uf[0]/alpha)
        u1.append(Uf[1]/alpha)

print("u0 : ",u0)
print("u1 : ",u1)

plt.imshow(frame)
plt.plot(u0, u1, 'rX')  # Projeter des points rouges en croix aux coordonnées spécifiées
plt.axis('off')  # Ne pas afficher les axes
plt.show()




# %% [markdown]
# Coordonnées de la caméra dans le repère monde

# %%
# page 24 du cours 
O1c_scalar = O1c.item()  # Extrait la valeur scalaire
O3c_scalar = O3c.item()  # Extrait la valeur scalaire

# Définir R et T_origin
R = np.array([[r11, r12, r13],
              [r21, r22, r23],
              [r31, r32, r33]])

# Créer T_origin en utilisant des scalaires
T_origin = np.array([O1c_scalar, O2c, O3c_scalar])  # Tableau 1D


R_trans = R.T
Pcam = -np.dot(R_trans,T_origin)
print('Pcam : ',Pcam)

# %% [markdown]
# Utilisation de OpenCV

# %%
print(type(bibliotheque['c_mm']))
print(bibliotheque['c_mm'])
objectPoints = [np.array(points, dtype=np.float32) for points in bibliotheque['c_mm']]
print(objectPoints)
imagePoints = [np.array(points, dtype=np.float32) for points in bibliotheque['c_ppx']]
#Taille de l'image
image_size = (frame.shape[1], frame.shape[0])

#Créer la matrice intrinsèque
mtx_init = np.array([[800, 0, frame.shape[1]/2],
                     [0, 800, frame.shape[0]/2],
                     [0, 0, 1]], dtype=np.float32)


#Appeler la fonction de calibration de la caméra
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints, imagePoints, image_size, None, None
) 

# Afficher les résultats
print("Matrice de la caméra : \n", mtx)
print("Coefficients de distorsion : \n", dist)
print("Erreur de reprojection : ", ret)


