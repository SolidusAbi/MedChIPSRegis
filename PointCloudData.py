import numpy as np
from matplotlib import pyplot as plt
import random
from scipy import ndimage

import cv2

class Data(object):
    '''
        Remember!! 
        In this class, reference points are represented by 0 because of distance transforms operations.
    '''
    def __init__(self, binary_img:np.array, n_ref_points: int):
        pts_ref = np.where(binary_img > 0)
        if len(pts_ref[0]) < n_ref_points:
            print("Number of reference points is bigger than the points from image. It will be reduced")
            n_ref_points = len(pts_ref[0]) 


        self.img = np.ones(binary_img.shape + (n_ref_points,))
        random_points = np.random.choice(len(pts_ref[0]), n_ref_points, replace=False)

        ref_points_coords = (pts_ref[0][random_points], pts_ref[1][random_points], np.arange(n_ref_points))
        self.img[ref_points_coords] = 0

    def getNumberOfRefPoints(self):
        return self.img.shape[2]

    def getPointCoords(self):
        '''
            Return the X, Y and Z coords in a list with each axes separated.
            The result is ordered  by Z coords.
            ([x_0, x_1, ..., x_n] [y_0, y_1, ..., y_n] [z_0, z_1, ..., z_n])
        '''
        # pts_coords = np.asarray(np.where( (self.img[:,:,0][:,:,np.newaxis] > 0 or  ) ))
        # print(self.img[:,:,0][:,:,np.newaxis].shape)
        # print(len(np.where(self.img[:,:,0][:,:,np.newaxis])[0]))
        # result = self.sortCoordsByDepth(pts_coords)
        x_coords = np.repeat(np.arange(self.img.shape[0]), self.img.shape[1]) 
        y_coords = np.arange(self.img.shape[1])
        y_coords = np.repeat(y_coords.reshape((1,) + y_coords.shape), self.img.shape[0], axis=0)
        y_coords = y_coords.ravel()
        z_coords = np.zeros(self.img.shape[0]*self.img.shape[1])
        return (x_coords, y_coords, z_coords) 

    def getRefPointCoords(self):
        '''
            Return the X, Y and Z coords in a list with each axes separated.
            The result is ordered  by Z coords.
            ([x_0, x_1, ..., x_n] [y_0, y_1, ..., y_n] [z_0, z_1, ..., z_n])
        '''
        find_pts = np.asarray(np.where(self.img < 1))
        result = self.sortCoordsByDepth(find_pts)
        return (result[0,:], result[1,:], result[2,:])

    def getOtherPointCoords(self):
        '''
            Return the X, Y and Z coords in a list with each axes separated.
            The result is ordered  by Z coords.
            ([x_0, x_1, ..., x_n] [y_0, y_1, ..., y_n] [z_0, z_1, ..., z_n])
        '''
        find_other_pts = np.asarray(np.where(self.img > 0))
        result = self.sortCoordsByDepth(find_other_pts)
        return (result[0,:], result[1,:], result[2,:])

    def sortCoordsByDepth(self, coords_array):
        '''
            This function is used in order to sort the differents coords by
            3rd axis order. 
            
            @params coords_array: array with the coords information where each axis
                is represented in specific row 
                [[X_0, .., X_n], 
                 [Y_0, .., Y_n], 
                 [Z_0, .., Z_n]])
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

    def getNormalizedRefPointsCoords(self):
        ref_coords = self.getRefPointCoords()
        x_dim, y_dim, _ = self.img.shape
        x_dim = x_dim/2
        y_dim = y_dim/2
        ref_coords = ((ref_coords[0]-x_dim)/x_dim, ((ref_coords[1]-y_dim)/y_dim))
        return ref_coords

    def normalizeCoords(self, coords):
        x_dim, y_dim, z_dim = self.img.shape
        x_dim = x_dim/2
        y_dim = y_dim/2
        z_dim = z_dim/2

        normalize_coords = coords.copy()
        normalize_coords[:, 0] = (normalize_coords[:, 0] - x_dim) / x_dim
        normalize_coords[:, 1] = (normalize_coords[:, 1] - y_dim) / y_dim
        normalize_coords[:, 2] = (normalize_coords[:, 2] - z_dim) / z_dim
        return normalize_coords

    def invNormalizeCoords(self, coords):
        x_dim, y_dim, z_dim = self.img.shape
        x_dim = x_dim/2
        y_dim = y_dim/2
        z_dim = z_dim/2

        inv_normalize_coords = coords.copy()
        inv_normalize_coords[:,:, 0] = (inv_normalize_coords[:,:, 0] * x_dim) + x_dim
        inv_normalize_coords[:,:, 1] = (inv_normalize_coords[:,:, 1] * y_dim) + y_dim
        inv_normalize_coords[:,:, 2] = (inv_normalize_coords[:,:, 2] * z_dim) + z_dim

        return inv_normalize_coords

class TargetData(Data):
    
    def __init__(self, binary_img:np.array, n_ref_points: int):
        super().__init__(binary_img, n_ref_points)
        self.interpolate()

    def interpolate(self):
        n_pixels = self.img.shape[0]*self.img.shape[1]
        self.dist_transform = ndimage.distance_transform_edt(self.img)
        self.alpha_dist_transform = np.zeros(self.dist_transform.shape)
        
        pts = self.getOtherPointCoords()   #Background points
        ref_pts = self.getRefPointCoords()  #Reference points

        denom = self.dist_transform[pts].reshape(n_pixels - 1, self.dist_transform.shape[2])
        denom = 1/denom[:]
        inverse_d = np.sum(denom[:], axis=0)  #There is a denominator for each slice

        w_dist = ((1/self.dist_transform[pts]).reshape(n_pixels - 1, self.dist_transform.shape[2]))/inverse_d

        self.alpha_dist_transform[pts] = w_dist.ravel()
        self.alpha_dist_transform[ref_pts] = 1

    def getDistanceTransform(self):
        return self.dist_transform

    def getAlphaDistanceTransform(self):
        return self.alpha_dist_transform

    def elasticTransform(self, v_transform):
        '''
            apply elastic transform to the point cloud data..
            @param v_transform: This matrix is normalized by image size 
        '''
        n_ref_points = self.img.shape[2]

        x_dim, y_dim, _ = self.img.shape
        x_dim = x_dim/2
        y_dim = y_dim/2
        #transform = (v_transform[:,0], v_transform[:,1], np.zeros(n_ref_points))

        transform = np.zeros((v_transform.shape[0], 3))
        transform[:,:-1] = v_transform
        
        v = np.zeros(((n_ref_points, self.img.shape[0], self.img.shape[1], 3)))
        for idx in range(n_ref_points):
            v[idx, :] = transform[idx]

        interpolated_transform = np.zeros(v.shape)
        alpha = self.getAlphaDistanceTransform()
        for slice_idx in range(n_ref_points):
            interpolated_transform[slice_idx,:] = alpha[:,:,slice_idx][:,:,np.newaxis] * v[slice_idx, :]

        test = self.getRefPointCoords()
        self.getNormalizedRefPointsCoords

        points_coords = self.changeCoordsFormat(self.getPointCoords())
        normalized_coords = self.normalizeCoords(points_coords)

        result = np.sum(interpolated_transform, axis=0) + normalized_coords.reshape((self.img.shape[0],self.img.shape[1],3)) 
        result = self.invNormalizeCoords(result)
        
        return result

    def changeCoordsFormat(self, coords_array):
        n_ref_points = len(coords_array[0])
        xyz_format = np.zeros((n_ref_points, 3))
        coords = np.asarray(coords_array)
        for idx in range(n_ref_points):
            xyz_format[idx] = coords.ravel()[idx::n_ref_points]

        return xyz_format

    def show(self):
        plt.imshow(self.img)
        plt.show()

if __name__ == "__main__":
    test_fixed = cv2.imread("test/synthetic_images/rect_fin.png", cv2.IMREAD_GRAYSCALE)
    test_moving = cv2.imread("test/synthetic_images/rect_ori.png", cv2.IMREAD_GRAYSCALE)
    test_fixed = cv2.Canny(test_fixed, 200, 255)
    test_moving = cv2.Canny(test_moving, 200, 255)
    data_fixed = Data(test_fixed, 20)
    data_moving = TargetData(test_moving, 41)

    print(data_moving.getNormalizedRefPointsCoords())