import numpy as np
from matplotlib import pyplot as plt
from math import ceil
from functools import reduce

class SimImage():
    '''
        This class generates a images with some 'noisy' points with a similar distance
        between them.
    '''
    def __init__(self, shape: tuple, n_ref_points: int):
        self.img = np.ones(reduce((lambda x, y: x * y), list(shape)) * n_ref_points)
        n_pixels = reduce((lambda x, y: x * y), list(shape))
        stride = (n_pixels/(n_ref_points-1)) 
        stride_2D = np.arange(0, n_pixels, step=stride).astype(np.uint)
        stride_3D = (np.arange(0, n_ref_points)*(n_pixels)).astype(np.uint)
        
        if len(stride_2D) < len(stride_3D):
            stride_2D = np.append(stride_2D, shape[0]*shape[1]-1)
            
        coords = stride_2D + stride_3D 
        
        self.img[coords.astype(np.uint)] = 0
        self.img = self.img.reshape(n_ref_points, shape[1], shape[0])
        
        #Transponse in order to get (W,H,Ch) format 
        self.img = np.transpose(self.img, axes=[2,1,0]).astype(np.float32)

    def getImage(self):
        return self.img.copy()

    def show(self, nCol = 3):
        '''
        Show the image using matplotlib
        @param nCol: number of columns of the figure.
        '''
        plt.figure()
        nRow = ceil(self.img.shape[2]/nCol)
        for point_idx in range(self.img.shape[2]):
            plt.subplot(nRow,nCol,point_idx+1)
            plt.imshow(self.img[:,:,point_idx], "gray")
            plt.xticks([]),plt.yticks([])
        plt.show()

if __name__ == "__main__":
    test = SimImage((24,24), 6)
    test.show()