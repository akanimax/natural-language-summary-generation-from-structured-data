from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input,Dense,LSTM
from keras.layers import Flatten
from keras.layers import Embedding,TimeDistributed

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
#print("field_content_words: ", field_content_words[0])
#print(len(field_content_words))
max_length = max([len(sample_pair) for sample_pair in field_content_words]) # Remark from Animesh: sample_pair is a bit confusing.
#print (max_length)
f.close()

# temporary correction:
# remove repeating field words from the list
# field_words = list(set(field_words))
############################################

#print content_words, len(content_words)
#print field_words, len(field_words)


######## prepare dataset for labels..labels as a list of sentences
file_nb =  open('../Data/train.nb','r')
num_sent = [] # info. of number of sentence per field-content pair.
for line in file_nb.readlines():
	nb_id = int(line.rstrip("\n"))
	num_sent.append(nb_id)
num_sentence = 0
for num in num_sent:
   num_sentence += num
# print num_sentence
file_sent =  open('../Data/train.sent','r')
labels = [] # list of output sentences.
labels_in = []
for line in file_sent.readlines():
	sent = line.rstrip("\n")
        sent = '\t ' + sent
	labels.append(sent)
        labels_in.append(sent)

# create proper alignment for data samples and levels pair
train_sample = []
for i,num in enumerate(num_sent):
   j = 0
   while (j < num):
      train_sample.append(field_content_words[i])
      j += 1

print train_sample

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
#print ((train_data_field[0:60]))

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

#print ((train_field_data[0]))
train_content_data = np.asarray(train_content_data)
train_field_data = np.asarray(train_field_data)
#print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2")
#print (train_field_data.shape)

## create training level data from level sentences

#print (labels[:3])
#find vocab length of level data and maximum level sequence length and id for each word in level set
label_vocab = []
def vocab_length(labels):
   temp = []
   for sent in labels:
      word_list = sent.split(" ")
      temp.append(len(word_list))
      for word in word_list:
         label_vocab.append(word)
   max_len = max(temp)
   return max_len
max_len_level = vocab_length(labels)
print (max_len_level)
#
encoded_level,level_dict,rev_level_dict,vocab_size_level = prepare_tokenizer(label_vocab)
#print (rev_level_dict)   #'professional': 15, 'the': 1, 'democratic': 119
#print ("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#print (vocab_size_level)
def level_data_encoded(labels):
   encoded_level_data = []
   for sent in labels:
      temp =[]
      word_list = sent.split(" ")
      for word in word_list[1:]:
         if (word in rev_level_dict):
            temp.append(rev_level_dict[word])
         else:
            temp.append(0)
      encoded_level_data.append(temp)
   return encoded_level_data
level_encoded = level_data_encoded(labels)
#print (level_encoded)

##### create one-hot vector for the level data

decoder_target_data = np.zeros(
    (num_sentence, max_len_level, vocab_size_level),
    dtype='float32')

for i,level_sent in enumerate(level_encoded):
   for j,sent in enumerate(level_sent):
      decoder_target_data[i][j][sent] = 1
#print (decoder_target_data.shape)

############### create decoder input data ...which is yt-1....

def level_data_encoded_in(labels):
   encoded_level_data = []
   for sent in labels:
      temp =[]
      word_list = sent.split(" ")
      temp.append(0)
      for word in word_list[1:]:
         if (word in rev_level_dict):
            temp.append(rev_level_dict[word])
         else:
            temp.append(0)
      while (len(temp) < max_len_level_in):
         temp.append(0)
      encoded_level_data.append(temp)

   return encoded_level_data


#print (labels_in[0:2])
max_len_level_in = vocab_length(labels_in)  # max sequence length
#print (max_len_level_in)
level_encoded_in = level_data_encoded_in(labels_in)
level_encoded_in = np.asarray(level_encoded_in)
#print ((level_encoded_in.shape))


###level_encoded_in,decoder_target_data,train_field_data,train_content_data................



## Define the model

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
encoder = LSTM(feature_dim,input_shape=(merged_embedding.shape[1],merged_embedding.shape[2]), return_sequences=True, return_state=True)
encoder_outputs, state_h, state_c = encoder(merged_embedding)


##### CREATE CONTENT BASED ATTENTION MECHANISM

# Decoder and dispather layer input from previous time step yt-1
dec_dis_input = Input(shape=(max_len_level_in,),dtype = 'int32',name = 'dec_dis_input')

# Embedding layer for yt-1
x_dec_dis_input = Embedding(output_dim=dim_embedding,input_dim=vocab_size_level,input_length=max_len_level_in)(dec_dis_input)

# feed this tensor to layer
field_mat = MyLayer_fcn_bias(dim_embedding)(x_dec_dis_input)
# print (field_mat)
########################
#layer of dot product between field embeddings and field_mat
field_weight = keras.layers.dot(inputs= [x_field,field_mat], axes=-1, normalize=False)

# Define similar matrix for content
content_mat = MyLayer_fcn_bias(feature_dim)(x_dec_dis_input)

# Define similar weight for content
content_weight = keras.layers.dot(inputs= [encoder_outputs,content_mat], axes=-1, normalize=False)

#print (content_weight)

##### Dummy part of the code to check the model is working or not
decoder_dense = Dense(vocab_size_level, activation='softmax')
decoder_outputs = decoder_dense(content_weight)


########## Dummy part end

model = Model(inputs = [field_input, content_input,dec_dis_input], outputs=decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([train_field_data,train_content_data,level_encoded_in], decoder_target_data,
          batch_size=1,
          epochs=1,
          validation_split=0.2)

# train_field_data => 23 x
# train_content_data =>
# level_encoded_in =>
