import numpy as np
import PointCloudData
import cv2
from matplotlib import pyplot as plt

from functools import partial
from pycpd import deformable_registration

def visualize(iteration, error, X, Y, ax):
    plt.cla()
    ax.scatter(X[:,0] ,  X[:,1], color='grey', label='Fixed image')
    ax.scatter(Y[:,0] ,  Y[:,1], color='green', label='Moving image')
    plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.01)

if __name__ == '__main__':
    fixed_img = cv2.imread('test/synthetic_images/rect_fin.png', cv2.IMREAD_GRAYSCALE)
    moving_img = cv2.imread('test/synthetic_images/rect_ori.png', cv2.IMREAD_GRAYSCALE)
    fixed_img = cv2.Canny(fixed_img, 200, 254)
    moving_img = cv2.Canny(fixed_img, 200, 254)


    n_points = 20
    fixed_pc = PointCloudData.Data(fixed_img, n_points)
    moving_pc = PointCloudData.TargetData(moving_img, n_points)

    _normalized_fixed_coords = np.asarray(fixed_pc.getNormalizedRefPointsCoords())
    _normalized_moving_coords = np.asarray(moving_pc.getNormalizedRefPointsCoords())

    normalized_fixed_coords = np.zeros((n_points, 2))
    normalized_moving_coords = np.zeros((n_points, 2))

    for idx in range(n_points):
        normalized_fixed_coords[idx] = _normalized_fixed_coords.ravel()[idx::20]
        normalized_moving_coords[idx] = _normalized_moving_coords.ravel()[idx::20]

    reg = deformable_registration(beta=0.1, **{ 'X': normalized_fixed_coords, 'Y': normalized_moving_coords, 'tolerance': 1e-10})

    # coords = [n_points, 2]

    # Visualize #
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    callback = partial(visualize, ax=fig.axes[0])

    reg.register(callback)
    #############

    gaussian_kernel, weight = reg.get_registration_parameters()

    moving_pc.elasticTransform(np.dot(gaussian_kernel, weight))
    # print(moving_pc.elasticTransform(np.dot(gaussian_kernel, weight)))
    # print(fixed_pc.getRefPointCoords())