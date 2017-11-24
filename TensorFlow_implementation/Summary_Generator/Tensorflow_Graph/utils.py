'''
    Library of helper tools for training and creating the Tensorflow graph of the
    system
'''

import numpy as np


def pad_sequences(seqs, pad_value = 0):
    '''
        funtion for padding the list of sequences and return a tensor that has all the sequences padded
        with leading 0s (for the bucketing phase)
        @param
        seqs => the list of integer sequences
        pad_value => the integer used as the padding value (defaults to zero)
        @return => padded tensor for this batch
    '''

    # find the maximum length among the given sequences
    max_length = max(map(lambda x: len(x), seqs))

    # create a list denoting the values with which the sequences need to be padded:
    padded_seqs = [] # initialize to empty list
    for seq in seqs:
        seq_len = len(seq) # obtain the length of current sequences
        diff = max_length - seq_len # calculate the padding amount for this seq
        padded_seqs.append(seq + [pad_value for _ in range(diff)])


    # return the padded seqs tensor
    return np.array(padded_seqs)
