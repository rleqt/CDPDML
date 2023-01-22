import numpy as np


def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

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
    # copy the initial data


    tmpsend = np.zeros_like(send)
    np.copyto(tmpsend, send)
    if send.size < size:
        np.append(tmpsend, np.zeros(size - send.size))

    tmprecv = np.zeros_like(tmpsend)
    # print(f'recv before is {tmprecv} and starts is {tmpsend}')
    np.copyto(tmprecv, tmpsend)
    # print(f'recv after is {tmprecv} and starts is {tmpsend}')
    buffer = np.zeros((tmprecv.size // size) + 1, dtype=recv.dtype)

    indexToSend = rank
    sizesarr = np.zeros(size)
    startarr = np.zeros(size)

    tmp_size_of_data = recv.size

    for i in range(size):
        if tmp_size_of_data % size != 0:
            sizesarr[i] = (tmp_size_of_data // size) + 1
            tmp_size_of_data -= 1
        else:
            sizesarr[i] = tmp_size_of_data // size
    for j in range(size):
        if j == 0:
            startarr[j] = 0
        else:
            startarr[j] = startarr[j-1] + sizesarr[j-1]

    # print(f'sizes is {sizesarr} and starts is {startarr}')

    for i in range(size - 1):
        # calculate the data you need to send

        # send the current data box to the next process
        last_request = comm.Isend(
            tmprecv.flat[int(startarr[indexToSend]):int(startarr[indexToSend] + sizesarr[indexToSend])], dest=((rank + 1) % size))
        # receive the latter data box from the latter process
        comm.Irecv(buffer, source=((rank - 1) % size)).wait()
        # calculate which data you received
        # print(buffer.size)
        # print(recv.size)
        # assign the data to the appropriate place in recv

        indexToSend = (indexToSend - 1) % size

        msg_len = sizesarr[indexToSend]  # need to be defined

        tmprecv.flat[int(startarr[indexToSend]):int(startarr[indexToSend] + sizesarr[indexToSend])] = op(tmprecv.flat[int(startarr[indexToSend]):int(startarr[indexToSend] + sizesarr[indexToSend])], buffer[0:int(msg_len)])  # add operation

        last_request.wait()

    indexToSend = (rank + 1) % size
    for i in range(size - 1):

        # calculate the data you need to send

        # send the current data box to the next process
        last_request = comm.Isend(tmprecv.flat[int(startarr[indexToSend]):int(startarr[indexToSend]+sizesarr[indexToSend])], dest=((rank + 1) % size))
        # receive the latter data box from the latter process
        comm.Irecv(buffer, source=((rank - 1) % size)).wait()
        # calculate which data you received
        # print(tmp_recv)

        # assign the data to the appropriate place in recv

        indexToSend = (indexToSend - 1) % size

        msg_len = sizesarr[indexToSend]  # need to be defined

        tmprecv.flat[int(startarr[indexToSend]):int(startarr[indexToSend]+sizesarr[indexToSend])] = buffer[0:int(msg_len)]

        last_request.wait()

    tmprecv.reshape(recv.size)
    recv.flat[0:recv.size] = tmprecv.flat
    # exit(1)
