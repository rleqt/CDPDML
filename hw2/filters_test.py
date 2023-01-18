import numpy as np
import matplotlib.pyplot as plt
import pickle
import timeit
from filters import *
from scipy.signal import convolve2d
import imageio
import os

# 7X7
edge_kernel = np.array([[-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18],
                        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
                        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
                        [-3/9, -2/4, -1/1, 0, 1/1, 2/4, 3/9],
                        [-3/10, -2/5, -1/2, 0, 1/2, 2/5, 3/10],
                        [-3/13, -2/8, -1/5, 0, 1/5, 2/8, 3/13],
                        [-3/18, -2/13, -1/10, 0, 1/10, 2/13, 3/18]])


flipped_edge_kernel = np.array([[3/18, 2/13, 1/10, 0, -1/10, -2/13, -3/18],
                        [3/13, 2/8, 1/5, 0, -1/5, -2/8, -3/13],
                        [3/10, 2/5, 1/2, 0, -1/2, -2/5, -3/10],
                        [3/9, 2/4, 1/1, 0, -1/1, -2/4, -3/9],
                        [3/10, 2/5, 1/2, 0, -1/2, -2/5, -3/10],
                        [3/13, 2/8, 1/5, 0, -1/5, -2/8, -3/13],
                        [3/18, 2/13, 1/10, 0, -1/10, -2/13, -3/18]])

# 5X5
blur_kernel = np.array([[1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/52, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25],
                        [1/25, 1/25, 1/25, 1/25, 1/25]])

# 3X3
shapen_kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])


def get_image(): 
    fname = 'data/lena.dat'
    f = open(fname, 'rb')
    lena = np.array(pickle.load(f))
    f.close()
    return np.array(lena[175:390, 175:390])


# Note: Use this on your local computer to better understand what the correlation does.
def show_image(image):
    """ Plot an image with matplotlib

    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()


def corr_comparison():
    """ Compare correlation functions run time.
    """
    image = get_image()

    def timer(kernel, f):
        return min(timeit.Timer(lambda: f(kernel, image)).repeat(10, 1))

    def correlation_cpu(kernel, image):
        return convolve2d(image, kernel, mode='same')

    print('CPU 3X3 kernel:', timer(shapen_kernel, correlation_cpu))
    print('Numba 3X3 kernel:', timer(shapen_kernel, correlation_numba))
    print('CUDA 3X3 kernel:', timer(shapen_kernel, correlation_gpu))
    print("---------------------------------------------")
    
    # a = correlation_cpu(blur_kernel, image)
    # b = correlation_gpu(blur_kernel, image)
    # print(np.array_equal(a,b))
    # indices = np.where(np.not_equal(a, b))
    # # Extract the different elements
    # different_elements = a[indices]
    # # Print the different elements
    # print(different_elements)
    # print(indices)

    # print(np.array_equal(correlation_cpu(blur_kernel, image),correlation_gpu(blur_kernel, image)))
    print('CPU 5X5 kernel:', timer(blur_kernel, correlation_cpu))
    print('Numba 5X5 kernel:', timer(blur_kernel, correlation_numba))
    print('CUDA 5X5 kernel:', timer(blur_kernel, correlation_gpu))
    print("---------------------------------------------")
    
    # print(np.array_equal(correlation_cpu(flipped_edge_kernel, image),correlation_gpu(edge_kernel, image)))
    print('CPU 7X7 kernel:', timer(flipped_edge_kernel, correlation_cpu))
    print('Numba 7X7 kernel:', timer(edge_kernel, correlation_numba))
    print('CUDA 7X7 kernel:', timer(edge_kernel, correlation_gpu))
    print("---------------------------------------------")


if __name__ == '__main__':
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda-9.0/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda-9.0/nvvm/libdevice/'
    corr_comparison()

    res = sobel_operator()
    # show_image(res)
    plt.imshow(res, cmap='gray')
    plt.savefig('filter4.png')
