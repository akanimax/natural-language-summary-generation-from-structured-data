'''
    Library of helper tools for preprocessing and converting the text data into
    compatible tensors
'''
# import all the required stuff
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

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
            if (pair[1] != "<none>"):
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



def prepare_tokenizer(words):
    '''
        funtion to generate vocabulary of the given list of words
        implemented by Anindya
        @param
        words => the list of words to be tokenized
    '''
    # obtain a tokenizer
    t = Tokenizer()
    t.fit_on_texts(words)
    field_dict = dict(); rev_field_dict = dict()

    for key,value in t.word_index.items():
        field_dict[value] = key
        rev_field_dict[key] = value

    vocab_size = len(t.word_index) + 1
	#print (vocab_size)
	# integer encode the documents
    encoded_docs = t.texts_to_sequences(words)

    # pad documents to a max length of max_length words
    padded_docs = pad_sequences(encoded_docs, maxlen=1, padding='post')
	#print(padded_docs)
    return padded_docs, field_dict,rev_field_dict, vocab_size
