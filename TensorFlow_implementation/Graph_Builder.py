
# coding: utf-8

# # In this Notebook, I'll write the script for building the Order-Planner Model defined in the base referenced paper.
# -------------------------------------------------------------------------------------------------------------------
# The jupyter allows for a very easy graph building process while using the tf.InteractiveSession(). It is almost as if we are using the eager execution strategy. [Note it is not exactly same as eager execution. we have to explicitly write <i> tenosr.eval() </i> for execution.]
# link to paper -> https://arxiv.org/abs/1709.00155
# 
# -------------------------------------------------------------------------------------------------------------------
# # Technology used: Tensorflow

# as usual, I'll start with the utility cells:

# In[1]:


# packages used for processing: 
import matplotlib.pyplot as plt # for visualization
import numpy as np

# for operating system related stuff
import os
import sys # for memory usage of objects
from subprocess import check_output

# The tensorflow_graph_package for this implementation
from Summary_Generator.Tensorflow_Graph.utils import *
from Summary_Generator.Text_Preprocessing_Helpers.pickling_tools import *

# import tensorflow temporarily:
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# to plot the images inline
get_ipython().magic(u'matplotlib inline')


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

# constants for this script
train_percentage = 90
learning_rate = 3e-4 # for learning rate -> https://twitter.com/karpathy/status/801621764144971776?lang=en
# I know the tweet was a joke, but I have noticed that this learning rate works quite well.


# ## Unpickle the processed data file and create the train_dev pratitions for it

# In[6]:


data = unPickleIt(plug_and_play_data_file)


# In[7]:


field_encodings = data['field_encodings']
field_dict = data['field_dict']

content_encodings = data['content_encodings']

label_encodings = data['label_encodings']
content_label_dict = data['content_union_label_dict']


# ## create a randomized cell that prints a complete sample to verify the sanity of the processed data

# In[8]:


total_samples = len(field_encodings)

random_index = np.random.randint(total_samples)

# extract the three parts of this random sample
random_field_sample = field_encodings[random_index]
content_sample = content_encodings[random_index]
label_sample = label_encodings[random_index]

# print the extracted sample in meaningful format
print("Table Contents: ")
print([(field_dict[field], content_label_dict[content]) 
       for (field, content) in zip(random_field_sample, content_sample)])

print("\n")
print("Summary: ")
print([content_label_dict[label] for label in label_sample])


# run the above cell multiple times to satisfy yourself that the data is still sane.

# ## Perform random shuffling of the input data

# In[9]:


X, Y = synch_random_shuffle_non_np(zip(field_encodings, content_encodings), label_encodings)


# ## Perform train_dev_splitting of the given data:

# In[10]:


train_X, train_Y, dev_X, dev_Y = split_train_dev(X, Y, train_percentage)


# In[11]:


train_X_field, train_X_content = zip(*train_X)
train_X_field = list(train_X_field); train_X_content = list(train_X_content)


# In[12]:


print("Number of Examples in Training set: ", len(train_X))
print("Number of Examples in the dev  set: ", len(dev_X))


# In[13]:


# Free up the resources by deleting non required stuff
del X, Y, field_encodings, content_encodings, train_X


# # Building graph here:

# Note, that the built graph will be later added to the code package Summary_Generator. This is being done here since the graph building process becomes quite easy with jupyter.

# step 0: Set the Hyper constants for the graph building process.

# I also put all the summary_ops along with the graph. While executing the graph we can decide whether we wish to generate the summary or not.

# In[14]:


# Set some hyper constants to be used in the graph building:

# random_seed value for consistent debuggable behaviour
seed_value = 3

# vocabulary sizes
field_vocab_size = data['field_vocab_size']
content_label_vocab_size = data['content_label_vocab_size']

# Embeddings size:
field_embedding_size = 100
content_label_embedding_size = 400 # This is a much bigger vocabulary compared to the field_name's vocabulary

# LSTM hidden state sizes
lstm_cell_state_size = hidden_state_size = 500 # they are same (for now)


# In[15]:


# graph reset point:
tf.reset_default_graph()


# step 1: Create placeholders for the computations in the graph

# In[16]:


# Placeholders for the input data:
with tf.variable_scope("Input_Data"):
    tf_field_encodings = tf.placeholder(tf.int32, shape=(None, None), name="input_field_encodings")
    tf_content_encodings = tf.placeholder(tf.int32, shape=(None, None), name="input_content_encodings")
    tf_label_encodings = tf.placeholder(tf.int32, shape=(None, None), name="input_label_encodings")
    
    # This is a placeholder for storing the lengths of the input sequences (they are padded to tensor)
    tf_input_seqs_lengths = tf.placeholder(tf.int32, shape=(None,), name="input_sequence_lengths")
    
    # This is a placeholder for storing the lengths of the decoder sequences (they are padded to tensor)
    tf_label_seqs_lengths = tf.placeholder(tf.int32, shape=(None,), name="decoder_sequence_lengths")


# In[17]:


# create the one-hot encoded values for the label_encodings
with tf.variable_scope("One_hot_encoder"):
    tf_one_hot_label_encodings = tf.one_hot(tf_label_encodings, depth=content_label_vocab_size)


# In[18]:


# check tf_field_encodings
print(tf_field_encodings)


# step 2: Obtain Embeddings for the input and the output sequences

# In[19]:


# Scope for the shared Content_Label matrix
with tf.variable_scope("Unified_Vocabulary_Matrix"):
    content_label_embedding_matrix = tf.get_variable("content_label_embedding_matrix", 
                                shape=(content_label_vocab_size, content_label_embedding_size), 
                                initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                                dtype=tf.float32)


# In[20]:


# Embeddings for the given input data:
with tf.variable_scope("Input_Embedder"):
    # Embed the field encodings:
    field_embedding_matrix = tf.get_variable("field_embedding_matrix", 
                                shape=(field_vocab_size, field_embedding_size), 
                                initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                                dtype=tf.float32)
    
    tf_field_embedded = tf.nn.embedding_lookup(field_embedding_matrix, tf_field_encodings, name="field_embedder")
    
    # Embed the content encodings: 
    
    
    tf_content_embedded = tf.nn.embedding_lookup(content_label_embedding_matrix, 
                                                 tf_content_encodings, name="content_embedder")


# In[21]:


print("Embedded_Input_Tensors: ", tf_field_embedded, tf_content_embedded)


# In[22]:


# Embeddings for the label (summary sentences):
with tf.variable_scope("Label_Embedder"):
    # embed the label encodings
    tf_label_embedded = tf.nn.embedding_lookup(content_label_embedding_matrix, 
                                                 tf_label_encodings, name="label_embedder")


# In[23]:


print("Embedded_Label_Tensors: ", tf_label_embedded)


# In[24]:


# Concatenate the Input embeddings channel_wise and obtain the combined input tensor
with tf.variable_scope("Input_Concatenator"):
    tf_field_content_embedded = tf.concat([tf_field_embedded, tf_content_embedded], axis=-1, name="concatenator")


# In[25]:


print("Final_Input_to_the_Encoder: ", tf_field_content_embedded)


# step 3: Create the encoder RNN to obtain the encoded input sequences. <b>(The Encoder Module)</b>

# In[26]:


with tf.variable_scope("Encoder"):
    encoded_input, encoder_final_state = tf.nn.dynamic_rnn (
                            cell = tf.nn.rnn_cell.LSTMCell(lstm_cell_state_size), # let all parameters to be default
                            inputs = tf_field_content_embedded,
                            sequence_length = tf_input_seqs_lengths,
                            dtype = tf.float32
                        )


# In[27]:


print("Encoded_vectors_bank for attention mechanism: ", encoded_input)


# In[28]:


# define the size parameter for the encoded_inputs
encoded_inputs_embeddings_size = encoded_input.shape[-1]
print encoded_inputs_embeddings_size


# In[29]:


print("Final_state obtained from the last step of encoder: ", encoder_final_state)


# step 4: define the Attention Mechanism for the Model <b>(The Dispatcher Module)</b>

# step 4.1: define the content based attention

# In[30]:


with tf.variable_scope("Content_Based_Attention/trainable_weights"):
    '''
        These weights and bias matrices must be compatible with the dimensions of the h_values and the f_values
        passed to the function below. If they are not, some exception might get thrown and it would be difficult
        to debug it. 
    '''
    # field weights for the content_based attention
    W_f = tf.get_variable("field_attention_weights", shape=(field_embedding_size, content_label_embedding_size),
                         initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value))
    b_f = tf.get_variable("field_attention_biases", shape=(field_embedding_size, 1),
                         initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value))
    
    # hidden states weights for the content_based attention
    W_c = tf.get_variable("content_attention_weights", 
                          shape=(encoded_inputs_embeddings_size, content_label_embedding_size),
                          initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value))
    b_c = tf.get_variable("content_attention_biases", shape=(encoded_inputs_embeddings_size, 1),
                          initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value))
    
    # Define the summary_ops for all the weights:
    W_f_summary = tf.summary.histogram("Content_based_attention/field_weights", W_f)
    b_f_summary = tf.summary.histogram("Content_based_attention/field_biases", b_f)
    W_c_summary = tf.summary.histogram("Content_based_attention/content_weights", W_c)
    b_c_summary = tf.summary.histogram("Content_based_attention/content_weights", b_c)


# In[31]:


with tf.variable_scope("Content_Based_Attention"):
    def get_content_based_attention_vectors(query_vectors):
        '''
            function that returns the alpha_content vector using the yt-1 (query vectors)
        '''
        # use the W_f and b_f to transform the query_vectors to the shape of f_values
        f_trans_query_vectors = tf.matmul(W_f, tf.transpose(query_vectors)) + b_f
        # use the W_c and b_c to transform the query_vectors to the shape of h_values
        h_trans_query_vectors = tf.matmul(W_c, tf.transpose(query_vectors)) + b_c
        
        # transpose and expand the dims of the f_trans_query_vectors
        f_trans_query_matrices = tf.expand_dims(tf.transpose(f_trans_query_vectors), axis=-1)
        # obtain the field attention_values by using the matmul operation
        field_attention_values = tf.matmul(tf_field_embedded, f_trans_query_matrices)
        
        # perform the same process for the h_trans_query_vectors
        h_trans_query_matrices = tf.expand_dims(tf.transpose(h_trans_query_vectors), axis=-1)
        hidden_attention_values = tf.matmul(encoded_input, h_trans_query_matrices)
        
        # Don't use the squeeze operation! that will cause the loss of shape information
        field_attention_values = field_attention_values[:, :, 0] # drop the last dimension (1 sized)
        hidden_attention_values = hidden_attention_values[:, :, 0] # same for this one
        
        # return the element wise multiplied values followed by softmax
        return tf.nn.softmax(field_attention_values * hidden_attention_values, name="softmax")


# step 4.2: define the link based attention

# In[32]:


with tf.variable_scope("Link_Based_Attention/trainable_weights"):
    '''
        The dimensions of the Link_Matrix must be properly compatible with the field_vocab_size.
        If they are not, some exception might get thrown and it would be difficult
        to debug it.
    '''
    Link_Matrix = tf.get_variable("Link_Attention_Matrix", shape=(field_vocab_size, field_vocab_size),
            dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.5, seed=seed_value))
    
    Link_Matrix_summary = tf.summary.histogram("Link_based_attention", Link_Matrix)


# In[33]:


print(Link_Matrix)


# In[34]:


# define the function for obtaining the link based attention values.
with tf.variable_scope("Link_Based_Attention"):
    def get_link_based_attention_vectors(prev_attention_vectors):
        '''
            This function generates the link based attention vectors using the Link matrix and the 
        '''
        # carve out only the relevant values from the Link matrix
        matrix_all_values_from = tf.nn.embedding_lookup(Link_Matrix, tf_field_encodings)
        
        # // TODO: Calculate the matrix_relevant_values from matrix_all_values_from
        matrix_relevant_values = tf.map_fn(lambda u: tf.gather(u[0],u[1],axis=1),
                                [matrix_all_values_from, tf_field_encodings], dtype=matrix_all_values_from.dtype)
        
        
        return tf.nn.softmax(tf.reduce_sum(tf.expand_dims(prev_attention_vectors, axis = -1) * 
                                           matrix_relevant_values, axis=1),name="softmax")


# step 4.3: define the hybrid attention

# In[35]:


# define the hybrid of the content based and the link based attention
with tf.variable_scope("Hybrid_attention/trainable_weights"):
    # for now, this is just the content_based attention:
    Zt_weights = tf.get_variable("zt_gate_parameter_vector", dtype=tf.float32,
                                 initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                                 shape=(hidden_state_size + field_embedding_size + content_label_embedding_size, 1))
    
    Zt_weights_summary = tf.summary.histogram("Hybrid_attention/zt_weights", Zt_weights)


# In[36]:


with tf.variable_scope("Hybrid_attention"):
    # define the hybrid_attention_calculator function:
    def get_hybrid_attention(h_values, y_values, content_attention, link_attention):
        '''
            function to calcuate the hybrid attention using the content_attention and the link_attention
        '''
        # calculate the e_f values
        e_t = tf.reduce_sum(tf.expand_dims(link_attention, axis=-1) * tf_field_embedded, axis=1)
        
        # create the concatenated vectors from h_values e_t and y_values
        input_to_zt_gate = tf.concat([h_values, e_t, y_values], axis=-1) # channel wise concatenation
        
        # perfrom the computations of the z gate:
        z_t = tf.nn.sigmoid(tf.matmul(input_to_zt_gate, Zt_weights))
        
        # calculate z_t~ value using the empirical values = 0.2z_t + 0.5
        z_t_tilde = (0.2 * z_t) + 0.5
        
        # compute the final hybrid_attention_values using the z_t_tilde values over content and link based values
        hybrid_attention = (z_t_tilde * content_attention) + ((1 - z_t_tilde) * link_attention)
        
        # return the calculated hybrid attention:
        return hybrid_attention


# step 5: create the decoder RNN to obtain the generated summary for the structured data <b>(The Decoder Module)</b>

# In[37]:


with tf.variable_scope("Decoder/trainable_weights"):
       # define the weights for the output projection calculation
       W_output = tf.get_variable(
                           "output_projector_matrix", dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                           shape=(hidden_state_size, content_label_vocab_size))
       b_output = tf.get_variable(
                           "output_projector_biases", dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                           shape=(content_label_vocab_size,))
       
       # define the weights and biases for the x_t calculation
       W_d = tf.get_variable(
                       "x_t_gate_matrix", dtype=tf.float32,
                       initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                       shape=((hidden_state_size + content_label_embedding_size), content_label_embedding_size))
       b_d = tf.get_variable(
                           "x_t_gate_biases", dtype=tf.float32,
                           initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                           shape=(content_label_embedding_size,))
       
       # define the summary ops for the defined weights and biases
       W_output_summary = tf.summary.histogram("Decoder/W_output", W_output)
       b_output_summary = tf.summary.histogram("Decoder/b_output", b_output)
       W_d_summary = tf.summary.histogram("Decoder/W_d", W_d)
       b_d_summary = tf.summary.histogram("Decoder/b_d", b_d)
       
       # create the LSTM cell to be used for decoding purposes
       decoder_cell = tf.nn.rnn_cell.LSTMCell(lstm_cell_state_size)


# In[38]:


def decode(start_tokens, mode = "inference", decoder_lengths = None, w_reuse = True):
    '''
        Function that defines the decoder op and returns the decoded sequence (the summary)
        
        @params:
        start_tokens = a tensor containing the start tokens (one for each sequence in the batch)
        mode = a value from "training" or "inference" to determine for how long the decoder rnn is to be unrolled.
               behaviour is as follows:
               "training" => The rnn will be unrolled until the max(decode_lengths). decode_lengths cannot be None.
               "inference" => decode_lengths is be ignored and unrolling will be done till <eos> is received
               
    '''
    with tf.variable_scope("Decoder", reuse = w_reuse):
        # define the function to obtain the predictions out of the given hidden_state_values
        def get_predictions(h_t_values):
            '''
                This function transforms the h_t_values into a one_hot_type probability vector
            '''
            # apply the output_projection gate to obtain the predictions from the h_t_values
            predictions = tf.matmul(h_t_values, W_output) + b_output
            
            # return the predictions:
            return predictions
        
        
        # define a function to obtain the values for the next input to the LSTM_cell (y_t values)
        def get_y_t_values(pred_vals):
            '''
                pred_vals = the tensor of shape [batch_size x content_label_vocab_size]
            '''
            
            # calculate the next words to be predicted 
            act_preds = tf.argmax(pred_vals, axis=-1)
            
            # perform embedding lookup for these act_preds
            y_t_values = tf.nn.embedding_lookup(content_label_embedding_matrix, act_preds)
            
            # return the calculated y_t_values
            return y_t_values
            
        
        # write the loop function for the raw_rnn:
        def decoder_loop_function(time, cell_output, cell_state, loop_state):
            '''
                The decoder loop function for the raw_rnn
                (In future will implement the attention mechanism using the loop_state parameter.)
                @params
                compatible with -> https://www.tensorflow.org/api_docs/python/tf/nn/raw_rnn
            '''
            if(cell_state is None):
                # initial call of the loop function
                finished = (time >= tf_label_seqs_lengths)
                next_input = start_tokens
                next_cell_state = encoder_final_state
                emit_output = tf.placeholder(tf.float32, shape=(content_label_vocab_size))
                next_loop_state = tf.zeros_like(tf_field_encodings, dtype=tf.float32)
                
            else:
                # we define the loop_state as the prev_hybrid attention_vector!
                prev_attention_vectors = loop_state # extract the prev_attention_vector from the loop state
                
                # obtain the predictions for the cell_output
                preds = get_predictions(cell_output)
                
                # obtain the y_t_values from the cell_output values:
                y_t_values = get_y_t_values(preds)
                
                ''' Calculate the attention: '''
                # calculate the content_based attention values using the defined module
                cont_attn = get_content_based_attention_vectors(y_t_values)
                
                # calculate the link based attention values
                link_attn = get_link_based_attention_vectors(prev_attention_vectors)
                # print "link_attention: ", link_attn
                
                # calculate the hybrid_attention
                hybrid_attn = get_hybrid_attention(cell_output, y_t_values, cont_attn, link_attn)
                
                ''' Calculate the x_t vector for next_input value'''
                # use the hybrid_attn to attend over the encoded_input (to calculate the a_t values)
                a_t_values = tf.reduce_sum(tf.expand_dims(hybrid_attn, axis=-1) * encoded_input, axis=1) 
                
                # apply the x_t gate
                x_t = tf.tanh(tf.matmul(tf.concat([a_t_values, y_t_values], axis=-1), W_d) + b_d)
                
                
                ''' Calculate the finished vector for perfoming computations '''
                # for now it is just the decoder length completed or not value.
                finished = (time >= decoder_lengths)
                
                ''' Copy mechanism is left (//TODO: change the following and implement copy mechanism)'''
                emit_output = preds
                
                # The next_input is the x_t vector so calculated:
                next_input = x_t
                # The next loop_state is the current hybrid_attention vectors
                next_loop_state = hybrid_attn
                # The next_cell_state is going to be equal to the cell_state. (we_don't tweak it)
                next_cell_state = cell_state
            
            # In both the cases, the return value is same.
            # return all these created parameters
            return (finished, next_input, next_cell_state, emit_output, next_loop_state)
        
        # use the tf.nn.raw_rnn to define the decoder computations
        outputs, _, _ = tf.nn.raw_rnn(decoder_cell, decoder_loop_function)
        
    # return the outputs obtained from the raw_rnn:
    return tf.transpose(outputs.stack(), perm=[1, 0, 2])


# step 6: define the training computations:

# In[39]:


with tf.name_scope("Training_computations"):
    outputs = decode(tf_label_embedded[:, 0, :], mode="training", 
                     decoder_lengths=tf_label_seqs_lengths, w_reuse=None)


# In[40]:


# print the outputs:
print("Output_tensor: ", outputs)


# step 7: define the cost function and the optimizer to perform the optimization on this graph.

# In[41]:


# define the loss (objective) function for minimization
with tf.variable_scope("Loss"):
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=tf_one_hot_label_encodings))
    
    # record the loss summary:
    loss_summary = tf.summary.scalar("Objective_loss", loss)


# In[42]:


# define the optimizer for this task:
with tf.variable_scope("Trainer"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # define the train_step for this:
    train_step = optimizer.minimize(loss)


# step _ : define the errands for the model

# In[43]:


with tf.variable_scope("Errands"): 
    init = tf.global_variables_initializer()
    all_summaries = tf.summary.merge_all()


# ## Create a stub_session to generate the graph visualization

# In[44]:


model_name = "Model_1"


# In[45]:


model_path = os.path.join(base_model_path, model_name)


# In[46]:


with tf.Session() as sess:
    tensorboard_writer = tf.summary.FileWriter(model_path, graph=sess.graph, filename_suffix=".bot")
    
    # initialize the session to generate the visualization file
    sess.run(init)
    
    tvars = tf.trainable_variables()
    tvars_vals = sess.run(tvars)
    
    for var, val in zip(tvars, tvars_vals):
        print(var.name)


# # Write the session runner to check if the training loops execute

# The following cell ensures that although there are no errors in the graph compilation, the runtime execution of the model also doesn't cause any problems

# In[47]:


# set the projector's configuration to add the embedding summary also:
conf = projector.ProjectorConfig()
embedding_field = conf.embeddings.add()
embedding_content_label = conf.embeddings.add()

# set the tensors to these embedding matrices
embedding_field.tensor_name = field_embedding_matrix.name
embedding_content_label.tensor_name = content_label_embedding_matrix.name

# add the metadata paths to these embedding_summaries:
embedding_field.metadata_path = os.path.join("..", "Metadata/fields.vocab")
embedding_content_label.metadata_path = os.path.join("..", "Metadata/content_labels.vocab")

# save the configuration file for this
projector.visualize_embeddings(tensorboard_writer, conf)


# In[48]:


''' The following is just a runtime checker session loop. This loop is not the training loop for the model.
Which is the reason why, the model is not saved upon executing'''

with tf.Session() as sess:
    # create a saver object:
    saver = tf.train.Saver(max_to_keep=3)
    
    if(os.path.isfile(os.path.join(model_path, "checkpoint"))):
        # load the saved weights:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
    else:
        # run the initializer to create the variables
        sess.run(init)
    
    # obtain the padded training data:
    inp_field = pad_sequences(train_X_field)
    inp_conte = pad_sequences(train_X_content)
    inp_label = pad_sequences(train_Y)
    # print inp_field.shape, inp_conte.shape, inp_label.shape
    
    # obtain the sequence lengths for the field_encodings and the label_encodings
    inp_lengths = get_lengths(train_X_field)
    lab_lengths = get_lengths(train_Y)
    # print inp_lengths, lab_lengths
    
    # run a loop for 1000 iterations:
    for epoch in range(1000):
        print "current_epoch: ", (epoch + 1)
        # execute the cost and the train_step
        predicts, _, cost = sess.run([outputs, train_step, loss], feed_dict = {
            tf_field_encodings: inp_field,
            tf_content_encodings: inp_conte,
            tf_label_encodings: inp_label,
            tf_input_seqs_lengths: inp_lengths,
            tf_label_seqs_lengths: lab_lengths
        })
        
        if((epoch + 1) % 10 == 0 or epoch == 0):
            # generate the summary for this batch:
            sums = sess.run(all_summaries, feed_dict = {
                tf_field_encodings: inp_field,
                tf_content_encodings: inp_conte,
                tf_label_encodings: inp_label,
                tf_input_seqs_lengths: inp_lengths,
                tf_label_seqs_lengths: lab_lengths
            })
            
            # save this generated summary to the summary file
            tensorboard_writer.add_summary(sums, global_step=(epoch + 1))
            
            # also save the model 
            saver.save(sess, os.path.join(model_path, model_name), global_step=(epoch + 1))
            
        print "Cost: ", cost, "\n\n"

