# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 11:59:48 2019

@author: MariaArmas
https://homepages.inf.ed.ac.uk/rbf/HIPR2/images/art5noi1.gif
https://homepages.inf.ed.ac.uk/rbf/HIPR2/images/art5dst2.gif
"""
import numpy as np
from functools import reduce
from math import ceil
from scipy import ndimage
import matplotlib.pyplot as plt
from write_vtk_file import write_unstructured_file
from pycpd import affine_registration

from simulation.images import SimImage


# =============================================================================
# 1. Crear mapa de distancias para cada punto
# 2. Transformada de distancia (scikit-image)
# 3. vi = sum(alpha) * vn = alpha' * vn
#    Matriz con tensores de cada punto
# 4. Aplicar la máscara 
# =============================================================================

def show(img):
    '''
    Show the image using matplotlib
    @param nCol: number of columns of the figure.
    '''
    nCol = 3
    plt.figure()
    nRow = ceil(img.shape[2]/nCol)
    for point_idx in range(img.shape[2]):
        plt.subplot(nRow,nCol,point_idx+1)
        plt.imshow(img[:,:,point_idx], "gray")
        plt.xticks([]),plt.yticks([])
    plt.show()


# n_ref_pts = 3
# shape = (5,5,n_ref_pts)

# def simulatedImage(shape):
#     sim_image = np.ones(reduce((lambda x, y: x * y), list(shape)))
#     stride = (shape[0]*shape[1]/(shape[2]-1)) 
#     stride_2D = np.arange(0, shape[0]*shape[1], step=stride).astype(np.uint)
#     stride_3D = np.arange(0, shape[2])*(shape[0]*shape[1])
    
#     if len(stride_2D) < len(stride_3D):
#         stride_2D = np.append(stride_2D, shape[0]*shape[1]-1)
        
#     coords = stride_2D + stride_3D 
    
#     sim_image[coords.astype(np.uint)] = 0
#     sim_image = sim_image.reshape(shape[2], shape[1], shape[0])
#     sim_image = np.transpose(sim_image, axes=[2,1,0]).astype(np.float32)
#     return(sim_image)

# sim_image = simulatedImage(shape)

'''
        New code 
'''
shape = (5,5)
n_ref_pts = 4
sim_image_object = SimImage(shape, n_ref_pts)
sim_image = sim_image_object.getImage()


#    def Interpolation(sim_image):
dist_transform, indices = ndimage.distance_transform_edt(sim_image,return_indices = True)

alpha_dist = np.zeros((dist_transform.shape[0],dist_transform.shape[1]))
n_pixels = dist_transform.shape[0]*dist_transform.shape[1]

#pts = np.where(sim_image > 0)
#ref_pts = np.where(sim_image < 1)

pts = sim_image_object.getOtherPointsCoords()
ref_pts = sim_image_object.getPointRefCoords()

denom = dist_transform[pts].reshape(n_pixels - 1, dist_transform.shape[2])
denom = 1/denom[:]
inverse_d = np.sum(denom[:], axis=0)  #Hay un denominador por cada plano

w_dist = ((1/dist_transform[pts]).reshape(n_pixels - 1, dist_transform.shape[2]))/inverse_d

alpha = np.zeros(dist_transform.shape)
alpha[pts] = w_dist.ravel()
alpha[ref_pts] = 1

# =============================================================================
# Importante: las cooordenadas de los ptos de ref no pueden ser manipuladas por otr pto de referencia
# =============================================================================
ref_x_coord, ref_y_coord, ref_z_coord = sim_image_object.getPointRefCoords()
displace_y_coord = np.asarray(ref_y_coord) + np.array([2,1,-1,-1]) #

displaced_image = np.ones(sim_image.shape)

displaced_pts = (ref_x_coord, displace_y_coord, ref_z_coord)
displaced_image[displaced_pts] = 0

displaced_pts = np.array(displaced_pts)
ref_pts = np.array(ref_pts)

disp_coords = displaced_pts - ref_pts
transforms = np.zeros((n_ref_pts,3))

for idx in range(n_ref_pts):
    transforms.ravel()[idx::3] = disp_coords.ravel()[idx::n_ref_pts]
    
    
#(npoints,dim,dim,[x y z])
v = np.zeros((n_ref_pts,shape[0],shape[1],3))
v[0,:,:,:] = transforms[0]
v[1,:,:,:] = transforms[1]
v[2,:,:,:] = transforms[2]

result = np.zeros(v.shape)
for slice_idx in range(n_ref_pts):
     result[slice_idx, :,:,:] = alpha[:,:,slice_idx][:,:,np.newaxis] * v[slice_idx,:,:,:]


m = sim_image.shape[0]
n = sim_image.shape[1]   
grey_values = sim_image[:,:,0]*sim_image[:,:,1]*sim_image[:,:,2]
   

Ym = np.zeros((m*n,3))  #Matriz de posiciones
r_values = np.zeros((n*m,1))    #Matriz de valores de intensidad
#g_values = np.zeros((n*m,1))
#b_values = np.zeros((n*m,1))
    
for y_idx in range(n):
    for x_idx in range(m):
        Ym[(x_idx *m)+y_idx, 0] = x_idx 
        Ym[(x_idx *m)+y_idx, 1] = y_idx
        r_values[(x_idx *m)+y_idx] = grey_values[x_idx ,y_idx]

#Posiciones finales
im_result = np.sum(result, axis=0) + Ym.reshape(5,5,3)



im_result0 = result[0,:,:,:] + Ym.reshape(5,5,3)
im_result1 = result[1,:,:,:] + Ym.reshape(5,5,3)
im_result2 = result[2,:,:,:] + Ym.reshape(5,5,3)

write_unstructured_file('grid.vtk',im_result.reshape(25,3),grey_values)
write_unstructured_file('grid0.vtk',im_result0.reshape(25,3),grey_values)
write_unstructured_file('grid1.vtk',im_result1.reshape(25,3),grey_values)
write_unstructured_file('grid2.vtk',im_result2.reshape(25,3),grey_values)

