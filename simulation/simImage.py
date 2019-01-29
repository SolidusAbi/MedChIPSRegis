import numpy as np

class SimImage():
    def __init__(self, shape: tuple, n_ref_points: int):
        self.img = np.ones(reduce((lambda x, y: x * y), list(shape)) * n_ref_points)
        stride = (shape[0]*shape[1]/(n_ref_points-1)) 
        stride_2D = np.arange(0, shape[0]*shape[1], step=stride).astype(np.uint)
        stride_3D = np.arange(0, n_ref_points)*(shape[0]*shape[1]).astype(np.uint)
        
        if len(stride_2D) < len(stride_3D):
            stride_2D = np.append(stride_2D, shape[0]*shape[1]-1)
            
        coords = stride_2D + stride_3D 
        
        self.img[coords.astype(np.uint)] = 0
        self.img = self.img.reshape(shape[2], shape[1], shape[0])
        self.img = np.transpose(self.img, axes=[2,1,0]).astype(np.float32)

    def getImage(self):
        return self.img()