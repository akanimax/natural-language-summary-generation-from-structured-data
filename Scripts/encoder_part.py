from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input,Dense,LSTM
from keras.layers import Flatten
from keras.layers import Embedding

#added for custom layer
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np


import re
import keras


from fcn_bias import MyLayer_fcn_bias

f = open('../Data/train.box','r')

#### Get the list of all field-content word pair from the input text..
field_content_words = []
field_words = []
content_words = []
def prepare_data():
   for line in f.readlines():
      temp = []
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

prepare_data()
print ("field_content_words: ", field_content_words[:10])
print ("length of the field_content_words' first example: ", len(field_content_words[0]))
max_length = max([len(sample_pair) for sample_pair in field_content_words])
print(max_length)
print(field_words)
print(content_words)
f.close()


######## prepare dataset for labels..labels as a list of sentences
file_nb =  open('../Data/train.nb','r')
num_sent = [] # info. of number of sentence per field-content pair.
for line in file_nb.readlines():
	nb_id = int(line.rstrip("\n"))
	num_sent.append(nb_id)

file_sent =  open('../Data/train.sent','r')
labels = [] # list of output sentences.
for line in file_sent.readlines():
	sent = line.rstrip("\n")
	labels.append(sent)

# create proper alignment for data samples and levels pair
train_sample = []
for i,num in enumerate(num_sent):
   j = 0
   while (j < num):
      train_sample.append(field_content_words[i])
      j += 1


# prepare tokenizer ,encoded sequence for field and content separately

def prepare_tokenizer(words):
	t = Tokenizer()
	t.fit_on_texts(words)
        field_dict = dict()
        rev_field_dict = dict()
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
        return padded_docs,field_dict,rev_field_dict,vocab_size

train_data_field,field_dict,rev_field_dict,vocab_size_field = prepare_tokenizer(field_words)
train_data_content,content_dict,rev_content_dict,vocab_size_content = prepare_tokenizer(content_words)
#print (vocab_size_content)
#print (field_dict)
print ((train_data_field[0:60]))

###create proper input datasets for training "train_field_data" and "train_content_data"
train_field_data = []
train_content_data = []
for sample in train_sample:
   temp_field = []
   temp_content = []
   for word_pair in sample:
      pair = word_pair.split(" ")
      #print (rev_field_dict[pair[0]])
      if pair[0] not in rev_field_dict:
         temp_field.append(0)
      else:
         temp_field.append(rev_field_dict[pair[0]])
      if pair[1] not in rev_content_dict:
         temp_content.append(0)
      else:
         temp_content.append(rev_content_dict[pair[1]])
   while (len(temp_field) < max_length):
      temp_field.append(0)
      temp_content.append(0)
   train_field_data.append(temp_field)
   train_content_data.append(temp_content)

print ((train_field_data[0]))

# input layer for field input
field_input = Input(shape=(max_length,),dtype = 'int32',name = 'field_input')

dim_embedding = 200
#embedding layer for field
x_field = Embedding(output_dim=dim_embedding,input_dim=vocab_size_field,input_length=max_length)(field_input)


#input layer for content input
content_input = Input(shape=(max_length,),dtype = 'int32',name = 'content_input')


#embedding layer for content
x_content = Embedding(output_dim=dim_embedding,input_dim=vocab_size_content,input_length=max_length)(content_input)

# we can then concatenate two embedding output
merged_embedding = keras.layers.concatenate([x_field,x_content],axis = -1)

# LSTM layer for encoder
feature_dim = 500
encoder = LSTM(feature_dim, return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder(merged_embedding)
#print (encoder_outputs.output_shape)
#model = Model(inputs = [field_input, content_input], outputs=stillnotcoded)


##### CREATE CONTENT BASED ATTENTION MECHANISM
#
# # Create a tensor of dimension vocab_size for level
# prev_output = ###need to fill
# # feed this tensor to layer
# field_mat = MyLayer_fcn_bias(dim_embedding)(prev_output)
#
# #layer of dot product between field embeddings and field_mat
# field_weight = keras.layers.dot(inputs = [x_field,field_mat], axes=1, normalize=False)
#
# # Define similar matrix for content
# content_mat = MyLayer_fcn_bias(feature_dim)(prev_output)
#
# # Define similar weight for content
# content_weight = keras.layers.dot(inputs= [encoder_outputs,content_mat], axes=1, normalize=False)
#
#
# #####TODO
# #DEFINE PREV_OUTPUT
