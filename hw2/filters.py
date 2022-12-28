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
    raise NotImplementedError("To be implemented")

@njit
def checkRange(image_row, image_col, i,j):
    return (i>=0 and i < image_row and j >= 0 and j < image_col)

@njit
def getNeighbors(kernel_row, kernel_col , i , j):
    middle_row = int((kernel_row-1)/2)
    middle_col = int((kernel_col-1)/2)
    return [(i+row_offset,j+col_offset) 
            for row_offset in prange(-1* middle_row, middle_row + 1)
            for col_offset in prange(-1* middle_col, middle_col + 1)]

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
    for i in prange(image_row):
        for j in prange(image_col):
            pixels = np.array([image[row,col] if checkRange(image_row, image_col, row,col) else 0 for (row,col) in getNeighbors(kernel_row, kernel_col, i, j)])
            pixels = pixels.reshape(kernel.shape)
            new_image[i,j] = np.sum(pixels * kernel)
    return new_image

def sobel_operator():
    '''Load the image and perform the operator
        ----------
        Return
        ------
        An numpy array of the image
        '''
    pic = load_image()
    sobel_filter1 = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_filter2 = np.array([[3, 0, -3],[10, 0, -10],[3, 0, -3]])
    sobel_filter3 = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    sobel_filter4 = np.array([[1,0,0,0,-1],[2,0,0,0,-2],[1,0,0,0,-1],[2,0,0,0,-2],[1,0,0,0,-1]])
    chosen_filter = sobel_filter1
    G_x = correlation_numba(chosen_filter, pic)
    G_y = correlation_numba(chosen_filter.T, pic)
    return np.sqrt((G_x ** 2) + (G_y ** 2))


def load_image(): 
    fname = 'data/image.jpg'
    pic = imageio.imread(fname)
    to_gray = lambda rgb : np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])
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
