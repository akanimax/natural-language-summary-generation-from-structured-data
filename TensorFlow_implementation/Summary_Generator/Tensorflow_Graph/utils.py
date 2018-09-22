'''
    Library of helper tools for training and creating the Tensorflow graph of the
    system
'''

import numpy as np

# Obtain the sequence lengths for the given input field_encodings / content_encodings (To feed to the RNN encoder)
def get_lengths(sequences):
    '''
        Function to obtain the lengths of the given encodings. This allows for variable length sequences in the
        RNN encoder.
        @param
        sequences = [2d] list of integer encoded sequences, padded to the max_length of the batch

        @return
        lengths = [1d] list containing the lengths of the sequences
    '''
    return list(map(lambda x: len(x), sequences))


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



# function to perform synchronous random shuffling of the training data
def synch_random_shuffle_non_np(X, Y):
    '''
        ** This function takes in the parameters that are non numpy compliant dtypes such as list, tuple, etc.
        Although this function works on numpy arrays as well, this is not as performant enough
        @param
        X, Y => The data to be shuffled
        @return => The shuffled data
    '''
    combined = list(zip(X, Y))

    # shuffle the combined list in place
    np.random.shuffle(combined)

    # extract the data back from the combined list
    X, Y = list(zip(*combined))

    # return the shuffled data:
    return X, Y



# function to split the data into train - dev sets:
def split_train_dev(X, Y, train_percentage):
    '''
        function to split the given data into two small datasets (train - dev)
        @param
        X, Y => the data to be split
        (** Make sure the train dimension is the first one)
        train_percentage => the percentage which should be in the training set.
        (**this should be in 100% not decimal)
        @return => train_X, train_Y, test_X, test_Y
    '''
    m_examples = len(X)
    assert train_percentage <= 100, "Train percentage cannot be greater than 100! NOOB!"
    partition_point = int((m_examples * (float(train_percentage) / 100)) + 0.5) # 0.5 is added for rounding

    # construct the train_X, train_Y, test_X, test_Y sets:
    train_X = X[: partition_point]; train_Y = Y[: partition_point]
    test_X  = X[partition_point: ]; test_Y  = Y[partition_point: ]

    assert len(train_X) + len(test_X) == m_examples, "Something wrong in X splitting"
    assert len(train_Y) + len(test_Y) == m_examples, "Something wrong in Y splitting"

    # return the constructed sets
    return train_X, train_Y, test_X, test_Y
