# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 18:32:13 2019

@author: Maria
""" 

import numpy as np
from matplotlib import pyplot as plt
import images
#from math import ceil
#from functools import reduce
from scipy import ndimage

class DistanceTransformType():
    euclidean = 0

class SimpleInterpolation():
    '''
        This class calculates alpha values that gives some 'weight' to the 
        pixels
    '''
    def __init__(self, sim_image, transform_type: DistanceTransformType):
        
        if transform_type == DistanceTransformType.euclidean: 
            None
            self.dist_transform, _ = ndimage.distance_transform_edt(sim_image,return_indices = True)
            self.alpha_dist = None
            self.n_pixels = None
            self.pts = None
            self.ref_pts = None
            self.denom = None
            self.inverse_d = None
            
        else:
            print('Error')
        
    def GetResult(self):
        self.alpha_dist = np.zeros((self.dist_transform.shape[0], self.dist_transform.shape[1]))
        self.n_pixels = self.dist_transform.shape[0]*self.dist_transform.shape[1]
        self.pts = sim_image_object.getOtherPointsCoords()
        self.ref_pts = sim_image_object.getPointRefCoords()
        self.denom = self.dist_transform[self.pts].reshape( self.n_pixels - 1, self.dist_transform.shape[2])
        self.denom = 1/ self.denom[:]
        self.inverse_d = np.sum(self.denom[:], axis=0)
        
        return self.alpha_dist
        
        
#dist_transform, indices = ndimage.distance_transform_edt(sim_image,return_indices = True)

#alpha_dist = np.zeros((dist_transform.shape[0],dist_transform.shape[1]))
#n_pixels = dist_transform.shape[0]*dist_transform.shape[1]
#
##pts = np.where(sim_image > 0)
##ref_pts = np.where(sim_image < 1)
#
#pts = sim_image_object.getOtherPointsCoords()
#ref_pts = sim_image_object.getPointRefCoords()

#denom = dist_transform[pts].reshape(n_pixels - 1, dist_transform.shape[2])
#denom = 1/denom[:]
#inverse_d = np.sum(denom[:], axis=0)

w_dist = ((1/dist_transform[pts]).reshape(n_pixels - 1, dist_transform.shape[2]))/inverse_d

alpha = np.zeros(dist_transform.shape)
alpha[pts] = w_dist.ravel()
alpha[ref_pts] = 1