import numpy as np


def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    np.copyto(recv, send)
    buffer = np.zeros(recv.size, dtype=recv.dtype)

    for thrd in range(size):
        # receive data from thrd and sum it with your data
        if thrd != rank:
            comm.Irecv(buffer, source=thrd).Wait()
            tmp = recv.flat[0:recv.size]
            recv.flat[0:recv.size] = op(tmp, buffer)
        # send everyone your data
        else:
            list = [((comm.Isend(send.flat[0:send.size], dest=j)) if thrd != j else None) for j in range(size)]
            for job in list:
                if job is not None:
                    job.Wait()




