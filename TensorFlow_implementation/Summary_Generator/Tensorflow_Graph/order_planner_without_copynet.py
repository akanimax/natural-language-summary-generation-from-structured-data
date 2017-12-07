'''
    This file generates the graph of the Model that we are going to use for the order planner for neural summary generator
    The function returns the graph object and some of the important handles of the tensors of the graph in a dictionary.
    Note, that all the possible tensor handles can be obtained by the tf.get_tensor_by_name() function. This is done to make
    things easy.
'''

import tensorflow as tf

# define the graph builder function:
def get_computation_graph(seed_value, field_vocab_size, content_label_vocab_size, field_embedding_size,
    content_label_embedding_size, lstm_cell_state_size, hidden_state_size, rev_content_label_dict):
    '''
        Function for building the graph for model 1:
        The architecture is same as defined in the base paper, except the copynet part
    '''
    graph = tf.Graph() # create a new graph object

    # define all the graph computations using the as_default function
    print("\n\n=============================================================================================================")
    print("Building the graph ... ")
    with graph.as_default():


        # ========================================================================
        # | Step 1:
        # ========================================================================

        print("\nstep 1: Creating input placeholders for the computations ...")
        # Placeholders for the input data:
        with tf.variable_scope("Input_Data"):
            tf_field_encodings = tf.placeholder(tf.int32, shape=(None, None), name="input_field_encodings")
            tf_content_encodings = tf.placeholder(tf.int32, shape=(None, None), name="input_content_encodings")
            tf_label_encodings = tf.placeholder(tf.int32, shape=(None, None), name="input_label_encodings")

            # This is a placeholder for storing the lengths of the input sequences (they are padded to tensor)
            tf_input_seqs_lengths = tf.placeholder(tf.int32, shape=(None,), name="input_sequence_lengths")

            # This is a placeholder for storing the lengths of the decoder sequences (they are padded to tensor)
            tf_label_seqs_lengths = tf.placeholder(tf.int32, shape=(None,), name="decoder_sequence_lengths")


        # create the one-hot encoded values for the label_encodings
        with tf.variable_scope("One_hot_encoder"):
            tf_one_hot_label_encodings = tf.one_hot(tf_label_encodings, depth=content_label_vocab_size)

        # print all placeholders for the encodings generated in step 1
        print("\tplaceholder for the field_encodings: ", tf_field_encodings)
        print("\tplaceholder for the content_encodings: ", tf_content_encodings)
        print("\tplaceholder for the label_encodings: ", tf_label_encodings)
        print("\tplaceholder for the input_sequence_lengths: ", tf_input_seqs_lengths)
        print("\tplaceholder for the label_sequence_lengths: ", tf_label_seqs_lengths)


        # ========================================================================
        # | Step 2:
        # ========================================================================

        print("\nstep 2: Creating Embeddings Mechanism for the input and the output words ...")
        # Scope for the shared Content_Label matrix
        with tf.variable_scope("Unified_Vocabulary_Matrix"):
            content_label_embedding_matrix = tf.get_variable("content_label_embedding_matrix",
                                        shape=(content_label_vocab_size, content_label_embedding_size),
                                        initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                                        dtype=tf.float32)

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


        print("\tEmbedded_Input_Tensors: ", tf_field_embedded, tf_content_embedded)

        # Embeddings for the label (summary sentences):
        with tf.variable_scope("Label_Embedder"):
            # embed the label encodings
            tf_label_embedded = tf.nn.embedding_lookup(content_label_embedding_matrix,
                                                         tf_label_encodings, name="label_embedder")

        print("\tEmbedded_Label_Tensors: ", tf_label_embedded)

        # Concatenate the Input embeddings channel_wise and obtain the combined input tensor
        with tf.variable_scope("Input_Concatenator"):
            tf_field_content_embedded = tf.concat([tf_field_embedded, tf_content_embedded], axis=-1, name="concatenator")

        print("\tFinal_Input_to_the_Encoder: ", tf_field_content_embedded)


        # ========================================================================
        # | Step 3:
        # ========================================================================

        print("\nstep 3: Creating the encoder RNN to obtain the encoded input sequences. (The Encoder Module) ... ")
        with tf.variable_scope("Encoder"):
            encoded_input, encoder_final_state = tf.nn.dynamic_rnn (
                                    cell = tf.nn.rnn_cell.LSTMCell(lstm_cell_state_size), # let all parameters to be default
                                    inputs = tf_field_content_embedded,
                                    sequence_length = tf_input_seqs_lengths,
                                    dtype = tf.float32
                                )
        print("\tEncoded_vectors_bank for attention mechanism: ", encoded_input)

        # define the size parameter for the encoded_inputs
        encoded_inputs_embeddings_size = encoded_input.shape[-1]

        print("\tFinal_state obtained from the last step of encoder: ", encoder_final_state)


        # ========================================================================
        # | Step 4:
        # ========================================================================

        print("\nstep 4: defining the Attention Mechanism for the Model (The Dispatcher Module) ...")


        print("**step 4.1: defining the content based attention")
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

                # drop the last dimension (1 sized)
                field_attention_values = tf.squeeze(field_attention_values, axis=[-1])
                hidden_attention_values = tf.squeeze(hidden_attention_values, axis=[-1])

                # return the element wise multiplied values followed by softmax
                return tf.nn.softmax(field_attention_values * hidden_attention_values, name="softmax")


        print("**step 4.2: defining the link based attention")
        with tf.variable_scope("Link_Based_Attention/trainable_weights"):
            '''
                The dimensions of the Link_Matrix must be properly compatible with the field_vocab_size.
                If they are not, some exception might get thrown and it would be difficult
                to debug it.
            '''
            Link_Matrix = tf.get_variable("Link_Attention_Matrix", shape=(field_vocab_size, field_vocab_size),
                    dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.5, stddev=0.5, seed=seed_value))

            Link_Matrix_summary = tf.summary.histogram("Link_based_attention", Link_Matrix)

        print("\tThe Link Matrix used for this attention: ", Link_Matrix)


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
                                                   matrix_relevant_values, axis=-1),name="softmax")


        print("**step 4.3: defining the hybrid attention")
        # define the hybrid of the content based and the link based attention
        with tf.variable_scope("Hybrid_attention/trainable_weights"):
            # for now, this is just the content_based attention:
            Zt_weights = tf.get_variable("zt_gate_parameter_vector", dtype=tf.float32,
                                         initializer=tf.random_uniform_initializer(minval=-1, maxval=1, seed=seed_value),
                                         shape=(hidden_state_size + field_embedding_size + content_label_embedding_size, 1))

            Zt_weights_summary = tf.summary.histogram("Hybrid_attention/zt_weights", Zt_weights)


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


        # ========================================================================
        # | Step 5:
        # ========================================================================

        print("\nstep 5: creating the decoder RNN to obtain the generated summary for the structured data (The Decoder Module) ...")

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
                        # define the fninshed parameter for the loop to determine whether to continue or not.
                        if(mode == "training"):
                            finished = (time >= decoder_lengths)

                        elif(mode == "inference"):
                            temp = tf.argmax(preds, axis=-1) # obtain the output predictions in encoded form
                            finished = (temp == rev_content_label_dict['<eos>'])

                        ''' Copy mechanism is left (//TODO: change the following and implement copy mechanism)'''
                        emit_output = preds

                        # The next_input is the x_t vector so calculated:
                        next_input = x_t
                        # The next loop_state is the current content_based attention
                        next_loop_state = cont_attn
                        # The next_cell_state is going to be equal to the cell_state. (we_don't tweak it)
                        next_cell_state = cell_state

                    # In both the cases, the return value is same.
                    # return all these created parameters
                    return (finished, next_input, next_cell_state, emit_output, next_loop_state)

                # use the tf.nn.raw_rnn to define the decoder computations
                outputs, _, _ = tf.nn.raw_rnn(decoder_cell, decoder_loop_function)

            # return the outputs obtained from the raw_rnn:
            return tf.transpose(outputs.stack(), perm=[1, 0, 2])


        # ========================================================================
        # | Step 6:
        # ========================================================================

        print("\nstep 6: defining the training computations ...")

        with tf.name_scope("Training_computations"):
            outputs = decode(tf_label_embedded[:, 0, :], mode="training",
                             decoder_lengths=tf_label_seqs_lengths, w_reuse=None)


        # print the outputs:
        print("\tFinal Output_Tensor obtained from the decoder: ", outputs)


        # ========================================================================
        # | Step 7:
        # ========================================================================

        print("\nstep 7: defining the cost function for optimization ...")

        # define the loss (objective) function for minimization
        with tf.variable_scope("Loss"):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=tf_one_hot_label_encodings))

            # record the loss summary:
            loss_summary = tf.summary.scalar("Objective_loss", loss)


        # ========================================================================
        # | Step 8:
        # ========================================================================

        print("\nstep 8: defining the computations for the inference mode ...")

        # define the computations for the inference mode
        with tf.variable_scope("inference_computations"):
            inf_outputs = decode(tf_label_embedded[:, 0, :])

        print("\tInference outputs: ", inf_outputs)


        # ========================================================================
        # | Step _:
        # ========================================================================

        print("\nstep _ : setting up the errands for TensorFlow ...")

        with tf.variable_scope("Errands"):
            all_summaries = tf.summary.merge_all()

    print("=============================================================================================================\n\n")

    # Generate the interface dictionary object for this defined graph
    interface_dict = {

        # Tensors for input placeholders into the graph
        "input": {
            "field_encodings": tf_field_encodings,
            "content_encodings": tf_content_encodings,
            "label_encodings": tf_label_encodings,
            "input_sequence_lengths": tf_input_seqs_lengths,
            "label_sequence_lengths": tf_label_seqs_lengths
        },

        # Tensors for embedding matrices:
        "field_embeddings": field_embedding_matrix,
        "content_label_embeddings": content_label_embedding_matrix,

        # Tensor for loass
        "loss": loss,

        # Tensor for the inference output:
        "inference": inf_outputs,

        # Tensor for training outputs
        "training_output": outputs,

        # Tensor for init and summary_ops
        "summary": all_summaries
    }

    # return the built graph object and it's interface dictionary:
    return graph, interface_dict
