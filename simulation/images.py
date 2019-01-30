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

    def getRefPointCoords(self):
        '''
            Return the X, Y and Z coords in a list with each axes separated.
            The result is ordered  by Z coords.
            [[x_0, x_1, ..., x_n] [y_0, y_1, ..., y_n] [z_0, z_1, ..., z_n]]
        '''
        print(np.asarray(np.where(self.img < 1)))
        find_pts = np.asarray(np.where(self.img < 1))
        n_ref_points = self.img.shape[2]
        print(len(find_pts[2]))
        print(n_ref_points)
        result = self.sortCoordsByDepth(find_pts)
        return result

    def getOtherPointCoords(self):
        '''
            Return the X, Y and Z coords in a list with each axes separated.
            The result is ordered  by Z coords.
            [[x_0, x_1, ..., x_n] [y_0, y_1, ..., y_n] [z_0, z_1, ..., z_n]]
        '''
        find_other_pts = np.asarray(np.where(self.img > 0))
        n_no_ref_points = (self.img.shape[0]*self.img.shape[1] - 1)*self.img.shape[2]
        print(len(find_other_pts[2]))
        print(n_no_ref_points)
        result = self.sortCoordsByDepth(find_other_pts)
        return result

    def sortCoordsByDepth(self, coords_array):
        '''
            This function is used in order to sort the differents coords by
            3rd axis order. 
            
            @params coords_array: array with the coords information where each axis
                is represented in specific row ([[X_0, .., X_n], [Y_0, .., Y_n], [Z_0, .., Z_n]])
        '''
        stride = len(coords_array[2])
        
        sorted_z = np.unique(np.sort(coords_array[2]))
        sorted_coords = np.zeros(coords_array.shape[0]*coords_array.shape[1]).astype(np.int64)
        
        result_idx = 0
        for z_idx in sorted_z.tolist():
            col_idx = np.argwhere(coords_array[2] == z_idx)
            coord_list = col_idx.ravel().tolist()
            
            for coord_idx in coord_list:
                sorted_coords[result_idx::stride] = coords_array.ravel()[coord_idx::stride]
                result_idx = result_idx + 1
       
        sorted_coords = sorted_coords.reshape(coords_array.shape)
        return sorted_coords
        

if __name__ == "__main__":
    test = SimImage((5,5), 4)
    test.show()