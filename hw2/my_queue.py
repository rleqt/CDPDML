import multiprocessing
from multiprocessing import Lock, Pipe

class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        self.reading_lock = multiprocessing.Lock()
        self.reading_pipe, self.writing_pipe = Pipe()
        self.pipe_size = multiprocessing.Value('i', 0)
        self.pipe_read = 0

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        self.reading_lock.acquire()
        try:
            self.pipe_size.value += 1
            self.writing_pipe.send(msg)
        finally:
            self.reading_lock.release()

    def get(self):
        '''Get the next message from queue (FIFO)

        Return
        ------
        An object
        '''
        ret = self.reading_pipe.recv()
        self.pipe_size.value -= 1
        return ret

    def length(self):
        '''Get the number of messages currently in the queue

        Return
        ------
        An integer
        '''
        return self.pipe_size.value
