# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 10:23:35 2018

@author: MariaArmas
cpd_affine

cpd source: https://github.com/siavashk/pycpd
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5432191
"""
from functools import partial
import matplotlib.pyplot as plt
from pycpd import affine_registration
import numpy as np
#import time
import SimpleITK as sitk
import cv2
import SkinSegmentation
from write_vtk_file import write_unstructured_file
#import os

output_path = '\\Users\MariaArmas\Documents\Registration\Hand\Images'

def myshow(img):
    nda = sitk.GetArrayViewFromImage(img)
    plt.figure()
    plt.imshow(nda)
    plt.axis('off')

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='grey', label='Fixed image')
    ax.scatter(Y[:,0] ,  Y[:,1], color='green', label='Moving image')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.01)
#    plt.savefig(os.path.join(output_path,'figure{:d}.jpg'.format(iteration)))    

#def main():
#Read and resize images
img1 = cv2.imread('mano2c.jpg')
img2 = cv2.imread('mano1c.jpg')

img1 = cv2.resize(img1,(480,480))
img2 = cv2.resize(img2,(480,480))

#Skin segmentation
sk1 = SkinSegmentation.MultiColorSpace(img1, SkinSegmentation.Format.RGB)
sk2= SkinSegmentation.MultiColorSpace(img2, SkinSegmentation.Format.RGB)
   
seg1 = sk1.apply()
seg2 = sk2.apply()

#Edge detection and sampling
canny1 =  cv2.Canny(seg1,200,255)
canny2 =  cv2.Canny(seg2,200,255)

plt.imshow(canny1)
plt.figure()
plt.imshow(canny2)
  
pc_1 = np.where(canny1[:] != 0)
pc_2 = np.where(canny2[:] != 0)
  
sampling1 = np.random.choice(len(pc_1[0]),1000)  #Buscar un muestreo mejor
sampling2 = np.random.choice(len(pc_2[0]),1000)  #Buscar un muestreo mejor

#Create txt with points cloud     
fd = open("point_cloud1.txt", 'w')
ux = canny1.shape[0]/2
uy = canny1.shape[1]/2

for idx in range(len(pc_1[0])):
    for i in range(len(sampling1)):
        if pc_1[0][idx] == sampling1[i]:
           fd.write("{} {}\n".format((pc_1[0][idx] - ux)/ux, (pc_1[1][idx] - uy)/uy))
    
fd = open("point_cloud2.txt", 'w')
ux = canny2.shape[0]/2
uy = canny2.shape[1]/2

for idx in range(len(pc_2[0])):
    for i in range(len(sampling2)):
        if pc_2[0][idx] == sampling2[i]:
            fd.write("{} {}\n".format((pc_2[0][idx] - ux)/ux, (pc_2[1][idx] - uy)/uy))

X = np.loadtxt('point_cloud1.txt')
Y = np.loadtxt('point_cloud2.txt')

y_copy = np.copy(Y)

#Registration
fig = plt.figure()
fig.add_axes([0, 0, 1, 1])
callback = partial(visualize, ax=fig.axes[0])

reg = affine_registration(**{ 'X': X, 'Y': Y }) 
reg.register(callback)
B, t  = reg.get_registration_parameters()

im = np.dot(y_copy, B) + np.tile(t, (y_copy.shape[0], 1))

plt.figure()
plt.cla()
plt.scatter(im[:,0],im[:,1])


#Apply to the real image

#Para aplicar la trasformacion a la imagen, hay que aplicarla punto a punto
#    T = B*ym + t, donde ym es de [2,1], y representa cada punto de la imagen
img = img2
RGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
RGB = RGB/255
r = RGB[:,:,0]
g = RGB[:,:,1]
b = RGB[:,:,2]

m = r.shape[0]
n = r.shape[1]         

Ym = np.zeros((m*n,2))  #Matriz de posiciones
r_values = np.zeros((n*m,1))    #Matriz de valores de intensidad
g_values = np.zeros((n*m,1))
b_values = np.zeros((n*m,1))


for y_idx in range(n):
    for x_idx in range(m):
        Ym[(y_idx *m)+x_idx, 0] = x_idx
        Ym[(y_idx *m)+x_idx, 1] = y_idx 
        r_values[(y_idx *m)+x_idx] = r[x_idx ,y_idx]
        g_values[(y_idx *m)+x_idx] = g[x_idx ,y_idx]
        b_values[(y_idx *m)+x_idx] = b[x_idx ,y_idx]
        
field_transform = np.dot(Ym, B) + np.tile(t, (Ym.shape[0], 1))

write_unstructured_file(Ym,r_values, g_values, b_values)
    




#    plt.figure()
#    ax=fig.axes[0]
#    plt.cla()
#    ax.scatter(X[:,0] ,  X[:,1], color='grey', label='Fixed image')
#    ax.scatter(im[:,0] ,  im[:,1], color='green', label='Moving image')

#if __name__ == '__main__':
#    main()
#    
