'''
    An OO implementation of the Model Object that composes with a Tensorflow_Graph and Optimizer
    to Train the network and eventually use the trained weights for predictions during the inference
'''

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
from Tensorflow_Graph.utils import *
import os
import numpy as np

# define the class for the Model
class Model:
    '''
        Model for training and Inferring on the task
        This is a simplified implementation of the Tensorflow's estimator module
    '''
    # Helper methods for initialization:
    def __setup_optimizer(self):
        # setup the optimizer with the graph
        with self.graph.as_default():
            # define the optimizer for this task:
        	with tf.variable_scope("Trainer"):
        	    # define the train_step for this:
        	    self.train_step = self.optimizer.minimize(self.loss)

        	with tf.variable_scope("Init"):
        		self.init = tf.global_variables_initializer()

    def __setup_graph(self):
        # print all the trainable variables for this graph
        with tf.Session(graph=self.graph) as sess:
            # initialize the session to generate the visualization file
            sess.run(self.init)

            tvars = tf.trainable_variables()
            tvars_vals = sess.run(tvars)

            print("\n\nAll the trainable variables in the Graph: ")
            for var, val in zip(tvars, tvars_vals):
                print(var.name)
            print("\n\n")

    def __get_tensorboard_writer(self, path):
        tensorboard_writer = tf.summary.FileWriter(path, graph=self.graph, filename_suffix=".bot")

        # set the projector's configuration to add the embedding summary also:
        conf = projector.ProjectorConfig()
        embedding_field = conf.embeddings.add()
        embedding_content_label = conf.embeddings.add()

        # set the tensors to these embedding matrices
        embedding_field.tensor_name = self.field_embedding_matrix.name
        embedding_content_label.tensor_name = self.content_label_embedding_matrix.name

        # add the metadata paths to these embedding_summaries:
        embedding_field.metadata_path = os.path.join("..", "Metadata/fields.vocab")
        embedding_content_label.metadata_path = os.path.join("..", "Metadata/content_labels.vocab")

        # save the configuration file for this
        projector.visualize_embeddings(tensorboard_writer, conf)

        # return the so created tensorboard_writer
        return tensorboard_writer

    # define the constructor of the graph
    def __init__(self, graph, interface_dict, optimizer, field_vocabulary, content_label_vocabulary):
        '''
            constructor of the class

            graph = The tensorflow graph of the network
            optimizer = The tensorflow optimizer object that is to be used for optimization
        '''
        self.graph = graph
        self.optimizer = optimizer

        # extract the parameters from the interface_dict:
        self.loss = interface_dict["loss"]
        self.all_summaries = interface_dict["summary"]
        self.outputs = interface_dict["training_output"]
        self.inference = interface_dict["inference"]

        self.inp_field_encodings = interface_dict["input"]["field_encodings"]
        self.inp_content_encodings = interface_dict["input"]["content_encodings"]
        self.inp_label_encodings = interface_dict["input"]["label_encodings"]
        self.inp_sequence_lengths = interface_dict["input"]["input_sequence_lengths"]
        self.lab_sequence_lengths = interface_dict["input"]["label_sequence_lengths"]

        self.field_embedding_matrix = interface_dict["field_embeddings"]
        self.content_label_embedding_matrix = interface_dict["content_label_embeddings"]

        # setup the vocabularies for the Model:
        self.field_vocabulary = field_vocabulary
        self.content_label_vocabulary = content_label_vocabulary

        self.__setup_optimizer()
        self.__setup_graph()

    # function for training the graph
    def train(self, X, Y, batch_size, no_of_epochs, checkpoint_factor, model_save_path, model_name):
        '''
            The training_function for the model.
        '''
        # Setup a tensorboard_writer:
        tensorboard_writer = self.__get_tensorboard_writer(model_save_path)

        ''' Start the actual Training loop: '''
        print("\n\nStarting the Training ... ")
        with tf.Session(graph=self.graph) as sess:
            # create a saver object:
            saver = tf.train.Saver(max_to_keep=3)

            # If old weights found, restart the training from there:
            checkpoint_file = os.path.join(model_save_path, "checkpoint")
            if(os.path.isfile(checkpoint_file)):
                # load the saved weights:
                saver.restore(sess, tf.train.latest_checkpoint(model_save_path))

                # load the global_step value from the checkpoint file
                with open(checkpoint_file, 'r') as checkpoint:
                    path = checkpoint.readline().strip()
                    global_step = int((path.split(':')[1]).split('-')[1][:-1])

            # otherwise initialize all the weights
            else:
                sess.run(self.init)

                # set the global_step to 0
                global_step = 0

            print("global_step: ", global_step)

            # run a loop for no_of_epochs iterations:
            for epoch in range(no_of_epochs):
                print("------------------------------------------------------------------------------------------------------------")
                print("current_epoch: ", (epoch + 1))

                # perform random shuffling on the training data:
                X, Y = synch_random_shuffle_non_np(zip(X[0], X[1]), Y)

                # unzip the shuffled X:
                X = zip(*X)

                # setup the data for training:
                # obtain the padded training data:
                train_X_field = X[0]; train_X_content = X[1]
                train_Y = Y; no_of_total_examples = len(train_X_field)

                # print len(train_X_field), len(train_X_content), len(train_Y)
                assert len(train_X_field) == len(train_X_content) and len(train_X_field) == len(train_Y), "input data lengths incompatible"

                # Iterate over the batches of the given train data:
                for batch_no in range(int(np.ceil(float(no_of_total_examples) / batch_size))):
                    # obtain the current batch of data:
                    start = (batch_no * batch_size); end = start + batch_size
                    batch_inp_field = train_X_field[start: end]
                    batch_inp_conte = train_X_content[start: end]
                    batch_inp_label = train_Y[start: end]
                    # pad the current batch of data:
                    inp_field = pad_sequences(batch_inp_field)
                    inp_conte = pad_sequences(batch_inp_conte)
                    inp_label = pad_sequences(batch_inp_label)
                    # extract the sequence lengths of examples in this batch
                    inp_lengths = get_lengths(batch_inp_field)
                    lab_lengths = get_lengths(batch_inp_label)


                    # execute the cost and the train_step
                    _, cost = sess.run([self.train_step, self.loss], feed_dict = {
                        self.inp_field_encodings: inp_field,
                        self.inp_content_encodings: inp_conte,
                        self.inp_label_encodings: inp_label,
                        self.inp_sequence_lengths: inp_lengths,
                        self.lab_sequence_lengths: lab_lengths
                    })
                    print "Range: ", "[", start, "-", (start + len(inp_field)), "]", " Cost: ", cost
                    global_step += 1

            	if((epoch + 1) % checkpoint_factor == 0 or epoch == 0):
                    # generate the summary for this batch:
                    sums, predicts = sess.run([self.all_summaries, self.outputs], feed_dict = {
                        self.inp_field_encodings: inp_field,
                        self.inp_content_encodings: inp_conte,
                        self.inp_label_encodings: inp_label,
                        self.inp_sequence_lengths: inp_lengths,
                        self.lab_sequence_lengths: lab_lengths
                    })

                    # save this generated summary to the summary file
                    tensorboard_writer.add_summary(sums, global_step=global_step)

                    # also save the model
                    saver.save(sess, os.path.join(model_save_path, model_name), global_step=global_step)

                    # print a random sample from this batch:
                    random_index = np.random.randint(len(inp_field))

                    random_label_sample = inp_label[random_index]
                    random_predicts_sample = np.argmax(predicts, axis = -1)[random_index]

                    # print the extracted sample in meaningful format
                    print("\nOriginal Summary: ")
                    print([self.content_label_vocabulary[label] for label in random_label_sample])

                    print("\nPredicted Summary: ")
                    print([self.content_label_vocabulary[label] for label in random_predicts_sample])

                print("------------------------------------------------------------------------------------------------------------")
        print("Training complete ...\n\n")
