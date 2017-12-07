from Summary_Generator.Tensorflow_Graph import order_planner_without_copynet
from Summary_Generator.Text_Preprocessing_Helpers.pickling_tools import *
from Summary_Generator.Tensorflow_Graph.utils import *
from Summary_Generator.Model import *
import numpy as np
import tensorflow as tf

# random_seed value for consistent debuggable behaviour
seed_value = 3

np.random.seed(seed_value) # set this seed for a device independant consistent behaviour

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





''' Name of the model:  '''
# This can be changed to create new models in the directory
model_name = "Model_1(without_copy_net)"

'''
	=========================================================================================================
	|| All Tweakable hyper-parameters
	=========================================================================================================
'''
# constants for this script
train_percentage = 90
batch_size = 3
checkpoint_factor = 10
learning_rate = 3e-4 # for learning rate -> https://twitter.com/karpathy/status/801621764144971776?lang=en
# I know the tweet was a joke, but I have noticed that this learning rate works quite well.

# Embeddings size:
field_embedding_size = 100
content_label_embedding_size = 400 # This is a much bigger vocabulary compared to the field_name's vocabulary

# LSTM hidden state sizes
lstm_cell_state_size = hidden_state_size = 500 # they are same (for now)
'''
	=========================================================================================================
'''





''' Extract and setup the data '''
# Obtain the data:
data = unPickleIt(plug_and_play_data_file)

field_encodings = data['field_encodings']
field_dict = data['field_dict']

content_encodings = data['content_encodings']

label_encodings = data['label_encodings']
content_label_dict = data['content_union_label_dict']
rev_content_label_dict = data['rev_content_union_label_dict']

# vocabulary sizes
field_vocab_size = data['field_vocab_size']
content_label_vocab_size = data['content_label_vocab_size']


X, Y = synch_random_shuffle_non_np(zip(field_encodings, content_encodings), label_encodings)

train_X, train_Y, dev_X, dev_Y = split_train_dev(X, Y, train_percentage)
train_X_field, train_X_content = zip(*train_X)
train_X_field = list(train_X_field); train_X_content = list(train_X_content)

# Free up the resources by deleting non required stuff
del X, Y, field_encodings, content_encodings, train_X






''' Obtain the TensorFlow graph of the order_planner_without_copynet Network'''
# just execute the get_computation_graph function here:
graph, interface_dict = order_planner_without_copynet.get_computation_graph (
	seed_value = seed_value,

	# vocabulary sizes
	field_vocab_size = field_vocab_size,
	content_label_vocab_size = content_label_vocab_size,

	# Embeddings size:
	field_embedding_size = field_embedding_size,
	content_label_embedding_size = content_label_embedding_size,

	# LSTM hidden state sizes
	lstm_cell_state_size = lstm_cell_state_size,
	hidden_state_size = hidden_state_size, # they are same (for now)
	rev_content_label_dict = rev_content_label_dict
)

''' Start the Training of the data '''
# Create the model and start the training on it
model_path = os.path.join(base_model_path, model_name)
model = Model(graph, interface_dict, tf.train.AdamOptimizer(learning_rate), field_dict, content_label_dict)
model.train((train_X_field, train_X_content), train_Y, batch_size, 100, checkpoint_factor, model_path, model_name)
