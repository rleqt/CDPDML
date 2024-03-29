from numba import njit, cuda, prange
import imageio
import matplotlib.pyplot as plt
import numpy as np


def correlation_gpu(kernel, image):
    '''Correlate using gpu
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''
    image_row = image.shape[0]
    image_col = image.shape[1]
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]

    # constructing the padded matrix
    padded = np.zeros((image_row + 2 * kernel_row, image_col + 2 * kernel_col), dtype=np.float64)
    padded[kernel_row:(kernel_row + image_row), kernel_col:(kernel_col + image_col)] = image
    padded_gpu = cuda.to_device(padded)
    kernel_gpu = cuda.to_device(kernel)

    new_image = np.zeros_like(image, dtype=np.float64)
    new_image_gpu = cuda.to_device(new_image)

    update_pixel_kernel[image_row, image_col](padded_gpu, kernel_gpu, new_image_gpu)
    return new_image_gpu.copy_to_host()


@cuda.jit
def update_pixel_kernel(padded, kernel, new_image):
    i = cuda.blockIdx.x
    j = cuda.threadIdx.x
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]

    # extracting a piece of the matrix with (i,j) in the middle of it
    middle_row, middle_col = int((-1) / 2), int((kernel_col - 1) / 2)
    pixels = padded[(i - middle_row):(i + middle_row + 1), (j - middle_col):(j + middle_col + 1)]

    # applying the kernel
    sum = 0
    for i in range(kernel_row):
        for j in range(kernel_col):
            sum += pixels[i, j] * kernel[i, j]
    new_image[i - kernel_row, j - kernel_col] = sum


@njit
def correlation_numba(kernel, image):
    '''Correlate using numba
    Parameters
    ----------
    kernel : numpy array
        A small matrix
    image : numpy array
        A larger matrix of the image pixels

    Return
    ------
    An numpy array of same shape as image
    '''
    new_image = np.zeros_like(image, dtype=np.float64)
    image_row = image.shape[0]
    image_col = image.shape[1]
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]
    # constructing the padded matrix
    padded = np.zeros((image_row + 2 * kernel_row, image_col + 2 * kernel_col), dtype=np.float64)
    padded[kernel_row:(kernel_row + image_row), kernel_col:(kernel_col + image_col)] = image
    for i in prange(kernel_row, image_row + kernel_row):
        for j in prange(kernel_col, image_col + kernel_col):
            # extracting a piece of the matrix with (i,j) in the middle of it
            middle_row, middle_col = int((kernel_row - 1) / 2), int((kernel_col - 1) / 2)
            pixels = padded[(i - middle_row):(i + middle_row + 1), (j - middle_col):(j + middle_col + 1)]
            # applying the kernel
            new_image[i - kernel_row, j - kernel_col] = np.sum(pixels * kernel)
    return new_image


def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    sobel_filter1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter2 = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])
    sobel_filter3 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_filter4 = np.array([[1, 0, 0, 0, -1], [2, 0, 0, 0, -2], [1, 0, 0, 0, -1], [2, 0, 0, 0, -2], [1, 0, 0, 0, -1]])
    correlation = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    chosen_filter = sobel_filter4
    G_x = correlation_numba(chosen_filter, pic)
    G_y = correlation_numba(chosen_filter.T, pic)
    return np.sqrt((G_x ** 2) + (G_y ** 2))


def load_image():
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray_pic = to_gray(pic)
    return gray_pic


def show_image(image):
    """ Plot an image with matplotlib
    Parameters
    ----------
    image: list
        2d list of pixels
    """
    plt.imshow(image, cmap='gray')
    plt.show()