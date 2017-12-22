
# coding: utf-8

# # In this Notebook, I'll preprocess the data and generate a plug_and_play pickle file for it
# -------------------------------------------------------------------------------------------------------------------
# # Technology used: basic preprocessing tools

# ### usual utility cells

# In[1]:


# packages used for processing: 
import numpy as np

# for operating system related stuff
import os
import sys # for memory usage of objects
from subprocess import check_output

# import the Text preprocessing helper to obtain the lists of field_name:content_word pairs
from Summary_Generator.Text_Preprocessing_Helpers.utils import *
from Summary_Generator.Tensorflow_Graph.utils import *
from Summary_Generator.Text_Preprocessing_Helpers.pickling_tools import *


# In[2]:


# Input data files are available in the "../Data/" directory.

def exec_command(cmd):
    '''
        function to execute a shell command and see it's 
        output in the python console
        @params
        cmd = the command to be executed along with the arguments
              ex: ['ls', '../input']
    '''
    print(check_output(cmd).decode("utf8"))


# In[3]:


# check the structure of the project directory
exec_command(['ls', '..'])


# In[4]:


np.random.seed(3) # set this seed for a device independant consistent behaviour


# In[5]:


''' Set the constants for the script '''

# various paths of the files
data_path = "../Data" # the data path

data_files_paths = {
    "table_content": os.path.join(data_path, "train.box"),
    "nb_sentences" : os.path.join(data_path, "train.nb"),
    "train_sentences": os.path.join(data_path, "train.sent")
}

base_model_path = "Models"
plug_and_play_data_file = os.path.join(data_path, "plug_and_play.pickle")

# constants for the preprocessing script
train_percentage = 95


# ## Extract the data from the related files and properly structure it

# In[6]:


field_content_words, field_words, content_words = prepare_input_data(data_files_paths['table_content'])


# In[7]:


# check if all the three lists are proper by printing them out
print("Field_content_words: ", field_content_words[1])
print("Field_words: ", field_words[:10])
print("Content_words: ", content_words[:10])


# In[8]:


# extract only the lenghts of the field_content_words and delete the field_content_words in order 
# to free up resources
pair_lengths = map(lambda x: len(x), field_content_words)
# print(pair_lengths)
del field_content_words


# In[9]:


label_sentences = prepare_input_labels(data_files_paths['nb_sentences'], data_files_paths['train_sentences'])


# In[10]:


# label_sentences are concatenated properly to obtain the decoder sentences.
for sent in label_sentences[:3]: print(sent + '\n')


# In[11]:


train_data_field, field_dict, rev_field_dict, vocab_size_field = prepare_tokenizer(field_words)


# In[ ]:


print(vocab_size_field, len(rev_field_dict), len(field_dict))
train_data_field[:3]


# In[ ]:


# use the group function to bring the data together:
field_seq = np.squeeze(train_data_field).tolist()
field_sequences = group_tokenized_sequences(field_seq, pair_lengths)


# In[ ]:


print field_dict


# In[ ]:


# print some slices of the field_sequences and the content_sequences:
print(field_sequences[:2])


# ## Check if the defined pad_sequences function works properly

# In[ ]:


padded_field_sequences = pad_sequences(field_sequences)
print("Length of padded_sequences: ", padded_field_sequences.shape)


# ## Perform structuring of the label_sentences and the content_words in order to create a unified vocabulary of it (for copy mechanism):

# Step 1: convert the label_sentences into a single flat list (order preserved) in order to tokenize it

# In[ ]:


# extract the length information from the label_sentences
label_sentences_lengths = map(lambda x: len(x.split()), label_sentences)
print(label_sentences_lengths[:3])


# In[ ]:


''' Warning: This is a huge map - reduce operation. And may take a long time to execute '''
label_words_list = reduce(lambda x,y: x + y, map(lambda x: x.split(), label_sentences))
print(label_words_list[:10])


# Step 2: store the lengths of the label_words_list and the content words in order to generate a unified vocabulary

# In[ ]:


content_words_label_words_split_point = len(content_words)


# In[ ]:


# concatenate the content_words and the label_words_list
unified_sequence = content_words + label_words_list
print("total length: ", len(unified_sequence))


# In[ ]:


# now use the tokenizer for this purpose:
temp, content_label_dict, rev_content_label_dict, vocab_size_content_label = prepare_tokenizer(unified_sequence)


# In[ ]:


# now again split the two lists separately and finally group them together to obtain the final stuff
content_seq = temp[: content_words_label_words_split_point]
label_seq = temp[content_words_label_words_split_point: ]


# In[ ]:


# use the group tokenized sequences function to restructure the tokenized input
content_seq = np.squeeze(content_seq).tolist()
label_seq = np.squeeze(label_seq).tolist()

content_sequences, label_sequences = (group_tokenized_sequences(content_seq, pair_lengths),
                                          group_tokenized_sequences(label_seq, label_sentences_lengths))


# ## Create the metadata file for the tensorboard_projector:

# In[ ]:


# metadata file path => Models/Metadata/
metadata_path = os.path.join(base_model_path, "Metadata")
print(metadata_path)


# In[ ]:


create_dot_vocab(field_dict, os.path.join(metadata_path, "fields.vocab"))


# In[ ]:


create_dot_vocab(content_label_dict, os.path.join(metadata_path, "content_labels.vocab"))


# ## Finally, perform the pickling of the Processed data

# In[ ]:


# create the structured dictionary to pickle in the pickle file:
pickling_data = {
    # ''' Input structured data: '''
    
    # field_encodings and related data:
    'field_encodings': field_sequences,
    'field_dict': field_dict,
    'field_rev_dict': rev_field_dict,
    'field_vocab_size': vocab_size_field,
    
    # content encodings and related data:
    'content_encodings': content_sequences,
    
    # ''' Label summary sentences: '''
    
    # label encodings and related data:
    'label_encodings': label_sequences,
    
    # V union C related data:
    'content_union_label_dict': content_label_dict,
    'rev_content_union_label_dict': rev_content_label_dict,
    'content_label_vocab_size': vocab_size_content_label
}


# use the function from this repository -> https://github.com/akanimax/machine-learning-helpers to perform pickling and unpickling. The code has been taken exactly and packaged in the Text_Preprocessing_Helpers module of this implementation

# In[ ]:


# pickle the above defined dictionary at the plug_and_play_data_file path
pickleIt(pickling_data, plug_and_play_data_file)


# Thus, the purpose of this notebook is now complete. We can directly use this pickled data and start building the tensorflow graph to go forward.

# ## See you in the graph building module! Asta la vista!
