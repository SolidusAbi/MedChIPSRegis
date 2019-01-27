# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 15:11:37 2018

@author: MariaArmas
cpd source: https://github.com/siavashk/pycpd
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5432191
"""
from functools import partial
import matplotlib.pyplot as plt
from pycpd import deformable_registration
from pycpd import affine_registration
import numpy as np
#import time
import SimpleITK as sitk
import cv2
import SkinSegmentation
from warp_images_tps import warp_images

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

fixed_image = cv2.resize(img1,(480,480))
moving_image = cv2.resize(img2,(480,480))

#cv2.imwrite('resized_img.jpg', moving_image)
#Skin segmentation
sk1 = SkinSegmentation.MultiColorSpace(fixed_image, SkinSegmentation.Format.RGB)
sk2= SkinSegmentation.MultiColorSpace(moving_image, SkinSegmentation.Format.RGB)
   
seg1 = sk1.apply()
seg2 = sk2.apply()

#Edge detection and sampling
canny_fixed =  cv2.Canny(seg1,200,255)
canny_moving =  cv2.Canny(seg2,200,250)

plt.imshow(canny_fixed)
plt.figure()
plt.imshow(canny_moving)
  
contour_fixed = np.where(canny_fixed[:] != 0)
contour_moving = np.where(canny_moving[:] != 0)
  
sampling_fixed = np.random.choice(len(contour_fixed[0]),1000)  #Buscar un muestreo mejor
sampling_moving = np.random.choice(len(contour_moving[0]),1000)  #Buscar un muestreo mejor

#Create the points clouds     
fd = open("point_cloud1.txt", 'w')
ux = canny_fixed.shape[0]/2
uy = canny_moving.shape[1]/2

for i in sampling_fixed:
   fd.write("{} {}\n".format(contour_fixed[0][i], contour_fixed[1][i]))
fd.close()    

fd = open("point_cloud2.txt", 'w')
ux = canny_moving.shape[0]/2
uy = canny_moving.shape[1]/2

#    for idx in range(len(pc_2[0])):
#        for i in range(len(sampling2)):
#            if pc_2[0][idx] == sampling2[i]:
#                fd.write("{} {}\n".format((pc_2[0][idx] - ux)/ux, (pc_2[1][idx] - uy)/uy))
for i in sampling_moving:
    fd.write("{} {}\n".format(contour_moving[0][i], contour_moving[1][i]))
fd.close()

X = np.loadtxt('point_cloud1.txt').astype(np.uint32)
Y = np.loadtxt('point_cloud2.txt').astype(np.uint32)

y_copy = np.copy(Y)
y_copy = (y_copy - ux)/ux

x_copy = np.copy(X)
x_copy = (x_copy - uy)/uy

#Registration
fig = plt.figure()
fig.add_axes([0, 0, 1, 1])
callback = partial(visualize, ax=fig.axes[0])

#Affine registration
#reg = affine_registration(**{ 'X': X, 'Y': Y }) 
#reg.register(callback)
#B, t  = reg.get_registration_parameters()
#
#registration_result = np.dot(y_copy, B) + np.tile(t, (y_copy.shape[0], 1))

#Deformable registration
reg = deformable_registration(**{ 'X': x_copy, 'Y': y_copy })
reg.register(callback)
gaussian_kernel, weight = reg.get_registration_parameters()

registration_result = y_copy + np.dot(gaussian_kernel, weight)


registration_result = ((registration_result*uy) + uy).astype(np.uint32)

#Bounding box
min = (np.min(registration_result[:, 0])-2)
#min = min/2
max = (np.max(registration_result[:, 0])+2)
#max = max/2
bbox = ((np.min(registration_result[:, 0]), np.min(registration_result[:, 1])), 
        (np.max(registration_result[:, 0]), np.max(registration_result[:, 1])),
        (np.max(registration_result[:, 0]), np.min(registration_result[:, 1])),
        (np.min(registration_result[:, 0]), np.max(registration_result[:, 1])))

    


#if __name__ == '__main__':
#    main()
#    
