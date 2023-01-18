import multiprocessing
from scipy import ndimage, misc
import numpy as np
import random


class Worker(multiprocessing.Process):

    def __init__(self, jobs, result, training_data, batch_size):
        super().__init__()

        ''' Initialize Worker and it's members.

        Parameters
        ----------
        jobs: JoinableQueue
            A jobs Queue for the worker.
        result: Queue
            A results Queue for the worker to put it's results in.
		training_data: 
			A tuple of (training images array, image lables array)
		batch_size:
			workers batch size of images (mini batch size)
        
        You should add parameters if you think you need to.
        '''
        self.jobs = jobs
        self.result = result

    @staticmethod
    def rotate(image, angle):
        '''Rotate given image to the given angle

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        angle : int
            The angle to rotate the image
            
        Return
        ------
        An numpy array of same shape
        '''
        # print("rotate")
        image_to_pass = image.reshape((28, 28))
        img_rotated = ndimage.rotate(image_to_pass, angle=angle, reshape=False)
        img_numpy = np.array(img_rotated)
        return img_numpy.flatten()

        # raise NotImplementedError("To be implemented")

    @staticmethod
    def shift(image, dx, dy):
        '''Shift given image

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        dx : int
            The number of pixels to move in the x-axis
        dy : int
            The number of pixels to move in the y-axis
            
        Return
        ------
        An numpy array of same shape
        '''
        image_to_pass = image.reshape((28, 28))
        return np.array( ndimage.shift( image_to_pass, (-dy, -dx), mode='constant', cval=0)).flatten()

    @staticmethod
    def add_noise(image, noise):
        '''Add noise to the image
        for each pixel a value is selected uniformly from the 
        range [-noise, noise] and added to it. 

        Parameters
        ----------
        image : numpy array
            An array of shape 784 of pixels
        noise : float
            The maximum amount of noise that can be added to a pixel

        Return
        ------
        An numpy array of same shape
        '''
        image_to_pass = image.reshape((28, 28))
        noised = image_to_pass + np.random.uniform(low=-noise, high=noise, size=image_to_pass.shape)
        maxed = np.maximum(noised, np.zeros_like(noised))
        mined = np.minimum(maxed, np.ones_like(maxed))
        return mined.flatten()

    @staticmethod
    def skew(image, tilt):
        '''Skew the image

        Parameters
        ----------
        image : numpy array
            An array of size 784 of pixels
        tilt : float
            The skew paramater

        Return
        ------
        An numpy array of same shape
        '''
        image_to_pass = image.reshape((28, 28))
        result = np.zeros((28, 28))
        for i in range(28):
            for j in range(28):
                index = int(j + i * tilt)
                if 0 <= index < 28:
                    result[i][j] = image_to_pass[i][index]
        return result.flatten()

    def process_image(self, image):
        '''Apply the image process functions
		Experiment with the random bounds for the functions to see which produces good accuracies.

        Parameters
        ----------
        image: numpy array
            An array of size 784 of pixels

        Return
        ------
        An numpy array of same shape
        '''
        iter = random.randint(0, 5)
        for i in range(iter):
            place = random.randint(0, 4)
            if place == 0:
                image = Worker.skew(image, 0.2 * random.random())
            elif place == 1:
                image = Worker.add_noise(image, 0.1)
            elif place == 2:
                image = Worker.rotate(image, int(10 * (random.random() - 0.5)))
            elif place == 3:
                image = Worker.shift(image, random.randint(-3, 3), random.randint(-3, 3))

        return image

    def run(self):
        '''Process images from the jobs queue and add the result to the result queue.
		Hint: you can either generate (i.e sample randomly from the training data)
		the image batches here OR in ip_network.create_batches
        '''
        cur_job = self.jobs.get()
        while cur_job is not None:
            final_image = (np.array([self.process_image(image) for image in cur_job[0]]), cur_job[1])
            self.result.put(final_image)
            self.jobs.task_done()
            cur_job = self.jobs.get()
        self.jobs.task_done()

