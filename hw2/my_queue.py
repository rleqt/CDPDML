
class MyQueue(object):

    def __init__(self):
        ''' Initialize MyQueue and it's members.
        '''
        raise NotImplementedError("To be implemented")

    def put(self, msg):
        '''Put the given message in queue.

        Parameters
        ----------
        msg : object
            the message to put.
        '''
        raise NotImplementedError("To be implemented")

    def get(self):
        '''Get the next message from queue (FIFO)
            
        Return
        ------
        An object
        '''
        raise NotImplementedError("To be implemented")
    
    def length(self):
        '''Get the number of messages currently in the queue
            
        Return
        ------
        An integer
        '''
        raise NotImplementedError("To be implemented")
