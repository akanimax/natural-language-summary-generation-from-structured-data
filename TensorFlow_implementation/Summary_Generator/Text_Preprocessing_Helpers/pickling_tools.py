from __future__ import print_function

import _pickle as pickle # pickle module in python
import os # for path related operations

'''
    Simple function to perform pickling of the given object. This fucntion may fail if the size of the object exceeds
    the max size of the pickling protocol used. Although this is highly rare, One might then have to resort to some other
    strategy to pickle the data.
    The second function available is to unpickle a file located at the specified path
'''

# coded by botman

# function to pickle an object
def pickleIt(obj, save_path):
    '''
        function to pickle the given object.
        @param
        obj => the python object to be pickled
        save_path => the path where the pickled file is to be saved
        @return => nothing (the pickle file gets saved at the given location)
    '''
    if(not os.path.isfile(save_path)):
        with open(save_path, 'wb') as dumping:
            pickle.dump(obj, dumping)

        print("The file has been pickled at:", save_path)

    else:
        print("The pickle file already exists: ", save_path)


# function to unpickle the given file and load the obj back into the python environment
def unPickleIt(pickle_path): # might throw the file not found exception
    '''
        function to unpickle the object from the given path
        @param
        pickle_path => the path where the pickle file is located
        @return => the object extracted from the saved path
    '''

    with open(pickle_path, 'rb') as dumped_pickle:
        obj = pickle.load(dumped_pickle)

    return obj # return the unpickled object
