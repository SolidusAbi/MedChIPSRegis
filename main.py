import numpy as np
import PointCloudData
import cv2
from matplotlib import pyplot as plt

from functools import partial
from pycpd import deformable_registration

from write_vtk_file import write_unstructured_file

from segmentation import skinSegmentation

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='grey', label='Fixed image')
    ax.scatter(Y[:,0] ,  Y[:,1], color='green', label='Moving image')
    plt.text(0.82, 0.92, 'Iteration: {:d}\nError: {:06.10f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.01)

def mariaCosas(coords, values):
    n_pixels = len(coords[0])

    gray_values = np.zeros((n_pixels, 1))
    for idx in range(n_pixels):
        gray_values[idx, 0] = values[coords[0][idx], coords[1][idx]]

    return gray_values

if __name__ == '__main__':

    
    # original_fixed_img = cv2.imread('test/synthetic_images/rect_fin.png', cv2.IMREAD_GRAYSCALE)
    # original_moving_img = cv2.imread('test/synthetic_images/rect_ori.png', cv2.IMREAD_GRAYSCALE)
    
    # original_fixed_img = cv2.resize(original_fixed_img,(120,120))
    # original_moving_img = cv2.resize(original_moving_img,(120,120))

    # canny_fixed_img = cv2.Canny(original_fixed_img, 200, 254)
    # canny_moving_img = cv2.Canny(original_moving_img, 200, 254)

    ##### Mano #####
    original_fixed_img = cv2.imread('test/mano2b.jpg')
    original_moving_img = cv2.imread('test/mano2b_deform.jpg')

    original_fixed_img = cv2.resize(original_fixed_img,(240,240))
    original_moving_img = cv2.resize(original_moving_img,(240,240))

    sk1 = skinSegmentation.MultiColorSpace(original_fixed_img, skinSegmentation.Format.RGB)
    sk2 = skinSegmentation.MultiColorSpace(original_moving_img, skinSegmentation.Format.RGB)

    original_fixed_img = sk1.apply()
    original_moving_img = sk2.apply()
    
    #Pasar a escala de grises
    original_moving_img = cv2.cvtColor(original_moving_img, cv2.COLOR_RGB2GRAY)
    original_fixed_img = cv2.cvtColor(original_fixed_img, cv2.COLOR_RGB2GRAY)

    canny_fixed_img =  cv2.Canny(original_fixed_img,200,254)
    canny_moving_img =  cv2.Canny(original_moving_img,200,254)
    ################


    n_points = 200
    fixed_pc = PointCloudData.Data(canny_fixed_img, n_points)
    moving_pc = PointCloudData.TargetData(canny_moving_img, n_points)

    _normalized_fixed_coords = np.asarray(fixed_pc.getNormalizedRefPointsCoords())
    _normalized_moving_coords = np.asarray(moving_pc.getNormalizedRefPointsCoords())

    normalized_fixed_coords = np.zeros((n_points, 2))
    normalized_moving_coords = np.zeros((n_points, 2))

    for idx in range(n_points):
        normalized_fixed_coords[idx] = _normalized_fixed_coords.ravel()[idx::n_points]
        normalized_moving_coords[idx] = _normalized_moving_coords.ravel()[idx::n_points]

    reg = deformable_registration(beta=2, **{ 'X': normalized_fixed_coords, 'Y': normalized_moving_coords, 'tolerance': 1e-10})

    # coords = [n_points, 2]

    # Visualize #
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg.register(callback)
    #############

    gaussian_kernel, weight = reg.get_registration_parameters()
    #Aplicar la transformación del CPD
    result = moving_pc.elasticTransform(np.dot(gaussian_kernel, weight))


    #Para que funcione lo de maria...
    result2 = result.reshape(result.shape[0]*result.shape[1], 3)

    gray_values = mariaCosas(moving_pc.getPointCoords(), original_moving_img)/255

    original_coords = np.transpose(np.asarray(fixed_pc.getPointCoords()))
    gray_values_fixed = mariaCosas(fixed_pc.getPointCoords(), original_fixed_img)/255


    write_unstructured_file('test.vtk', result2, original_moving_img.shape, gray_values)
    write_unstructured_file('fixed.vtk', original_coords, original_fixed_img.shape, gray_values_fixed)
    
    # moving_pc.elasticTransform(np.dot(gaussian_kernel, weight))
    # print(fixed_pc.getRefPointCoords())