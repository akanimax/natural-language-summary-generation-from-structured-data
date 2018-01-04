'''
    Library of helper tools for preprocessing and converting the text data into
    compatible tensors
'''

# import all the required stuff
from __future__ import print_function
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os # for the metadata file creator function

def prepare_input_data(table_file_path):
    '''
        function for obtaining the appropriate lists of words from the text file
        implemented by Anindya
        @param
        table_file_path => the path pointing to the location of the table file
        @return
        field_content_words, field_words, and content_words
    '''
    # obtain all the lines from the text file
    data_lines = open(table_file_path, 'r').readlines()

    # initialize all the lists to empty ones
    field_content_words = []; field_words = []; content_words = []

    # iterate over each line:
    for line in data_lines:
        temp = [] # create an empty array
        element_list = line.rstrip("\n").split("\t")
        for element in element_list:
            pair = element.split(":")

            '''
                small modification from Animesh:
                don't filter out the <none> pairs. Use all the data
            '''

            pair[0] = re.sub(r'[0-9]+', '', pair[0]).replace("_","")
            pair[1] = pair[1].split(" ")
            for word in pair[1]:
                temp_word = (pair[0] + " " + word)
                temp.append(temp_word)
                content_words.append(word)
                field_words.append(pair[0])
        field_content_words.append(temp)

    # remove the repeating field names from the list
    # field_words = list(set(field_words))
    # don't do this

    # return the so constructed lists
    return field_content_words, field_words, content_words



def prepare_input_labels(nb_file_path, sent_file_path):
    '''
        function for concatenating the label sentences in the file according to the numbers.
        implemented by Animesh
        @param
        nb_file_path => the path pointing to the location of the train.nb file
        sent_file_path => the path pointing to the location of the train.sent file
        @return
        aligned_labels_sequences
    '''
    # generate lists of numbers and raw label_sentences
    nums = open(nb_file_path, 'r').readlines(); sents = open(sent_file_path, 'r').readlines()

    # make the nums integers
    nums = map(int, nums);

    # make sure the number of sentences and the nums match
    assert sum(nums) == len(sents), "Length mismatch between the train.nb and train.sent files"

    # run a simple loop to concatenate sentences belonging to one single training example
    label_sents = [] # initialize to empty list
    for num in nums:
        count = 0; sent = '<start>' # initialize counter and sentence
        while count < num:
            sent += ' ' + sents.pop(0).strip(); count += 1
        # add the sentence to the label_sents list
        label_sents.append(sent + ' <eos>')

    # return the aligned_labels_sequences
    return label_sents



def prepare_tokenizer(words, max_word_length = None):
    '''
        funtion to generate vocabulary of the given list of words
        implemented by Anindya
        @param
        words => the list of words to be tokenized
    '''
    # flatten the words list:
    print("flattening the words into a single sequence ... ")
    flat_words = []; # initialize to empty list
    for i in range(len(words)):
        flat_words += words[i]
        if(i % 10000 == 0):
            print("joined", i, "examples")

    # obtain a tokenizer
    print("\nmaximum words to work with: ", max_word_length)
    t = Tokenizer(num_words = max_word_length, filters = '') # don't let keras ignore any words
    print("\nKeras's tokenizer kicks off ... ")
    t.fit_on_texts(flat_words)
    field_dict = dict(); rev_field_dict = dict()

    print("\nbuilding the dict and the rev_dict ... ")
    if(max_word_length is not None):
        vals = t.word_index.items()
        vals = sorted(vals, key=lambda x: x[1])
        for key,value in vals[:max_word_length - 1]:
            field_dict[value] = key
            rev_field_dict[key] = value
    else:
        for key,value in t.word_index.items():
            field_dict[value] = key
            rev_field_dict[key] = value


    ''' Small modification from Animesh
        # also add the '<unk>' token to the dictionary at 0th position
    '''
    field_dict[0] = '<unk>'; rev_field_dict['<unk>'] = 0


    print("\nencoding the words using the dictionary ... ")
    for i in range(len(words)):
        for j in range(len(words[i])):
            if(words[i][j] in rev_field_dict):
                words[i][j] = rev_field_dict[words[i][j]]
            else:
                words[i][j] = rev_field_dict['<unk>']

        if(i % 10000 == 0):
            print("encoded", i, "examples")

    vocab_size = len(field_dict)
    return words, field_dict, rev_field_dict, vocab_size


def group_tokenized_sequences(flat_seq, lengths):
    '''
        funtion to group the seqs together to original form after tokenization
        implemented by Animesh
        @param
        flat_seq => flat list of words (in order)
        lengths => list of lengths of each sequence in the dataset
        @return => grouped_seq
    '''

    # check if the lengths and the field_seq and the content_seq lengths are compatible
    assert sum(lengths) == len(flat_seq), "Lengths are not compatible"

    # perform the grouping:
    grouped_seqs = [] # initialize to empty list
    for length in lengths:
        count = 0; temp_grouped_seq = [] # initialize counter and storer list
        while(count < length):
            temp_grouped_seq.append(flat_seq.pop(0))
            count += 1

        # add the so contructed lists to the main groupings
        grouped_seqs.append(temp_grouped_seq)

    # finally return the so created lists
    return grouped_seqs


# create a function to generate a file for the given dictionary in the .vocab format
def create_dot_vocab(save_dict, path):
    '''
        Function for creating the Metadata file for the given vocab dict at the specified path

        @params:
        save_dict = the dictionary object used to create the metadata file
        path = the path where this dictionary is to be saved

        @return:
        None
    '''
    if(os.path.isfile(path)):
        print("The file already exists: ", path)

    else:
        with open(path, 'w') as metadatat_file:
            for (_, value) in save_dict.items():
                metadatat_file.write(value + "\n")
        # print a message that the file has been generated after completion
        print("The file has been created at: ", path)
