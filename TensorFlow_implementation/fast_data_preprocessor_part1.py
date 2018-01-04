''' 
    script for preprocessing the data from the files
    This script is optimized for producing the processed data faster
'''
from __future__ import print_function
import numpy as np 
import os
from Summary_Generator.Text_Preprocessing_Helpers.pickling_tools import *

# set the data_path
data_path = "../Data"

data_files_paths = {
    "table_content": os.path.join(data_path, "train.box"),
    "nb_sentences" : os.path.join(data_path, "train.nb"),
    "train_sentences": os.path.join(data_path, "train.sent")
}

# generate the lists for all the samples in the dataset by reading the file once


#=======================================================================================================================
# Read the file for field_names and content_names
#=======================================================================================================================


print("Reading from the train.box file ...")
with open(data_files_paths["table_content"]) as t_file:
    # read all the lines from the file:
    table_contents = t_file.readlines()

    # split all the lines at tab to generate the list of field_value pairs
    table_contents = map(lambda x: x.strip().split('\t'), table_contents)


print("splitting the samples into field_names and content_words ...")
# convert this list of string pairs into list of lists of tuples
table_contents = map(lambda y: map(lambda x: tuple(x.split(":")), y), table_contents)

# write a loop to separate out the field_names and the content_words
count = 0; field_names = []; content_words = [] # initialize these to empty lists
for sample in table_contents:
    # unzip the list:
    fields, contents = zip(*sample)

    # modify the fields to discard the _1, _2 labels
    fields = map(lambda x: x.split("_")[0], fields)

    # append the lists to appropriate lists
    field_names.append(list(fields)); content_words.append(list(contents))

    # increment the counter
    count += 1

    # give a feed_back for 1,00,000 samples:
    if(count % 100000 == 0):
         print("seperated", count, "samples")

print("\nfield_names:\n", field_names[: 3], "\n\ncontent_words:\n", content_words[: 3])



#==================================================================================================================
# Read the file for the labels now
#==================================================================================================================
print("\n\nReading from the train.nb and the train.sent files ...")
(labels, label_lengths) = (open(data_files_paths["train_sentences"]), open(data_files_paths["nb_sentences"]))
label_words = labels.readlines(); lab_lengths = label_lengths.readlines()
# close the files:
labels.close(); label_lengths.close()

print(label_words[: 3])

# now perfrom the map_reduce operation to receive the a data structure similar to the field_names and content_words
print("grouping lines in train.sent according to the train.nb ... ")
count = 0; label_sentences = [] # initialize to empty list

for length in lab_lengths:
    temp = []; cnt = 0;
    while(cnt < int(length)):
        sent = label_words.pop(0)        
        # print("sent", sent)
        temp += sent.strip().split(' ') 
        cnt += 1
        # print("temp ", temp)

    # append the temp to the label_sentences
    label_sentences.append(temp)

    # increment the counter
    count += 1

    # print a feedback for 1000 samples:
    if(count % 1000 == 0):
        print("grouped", count, "label_sentences")


print(label_sentences[-3:])


print("pickling the stuff generated till now ... ")
# finally pickle the objects into a temporary pickle file:
# temp_pickle object definition:
temp_pickle = {
    "fields": field_names,
    "content": content_words,
    "label": label_sentences
}

pickleIt(temp_pickle, "temp.pickle")
