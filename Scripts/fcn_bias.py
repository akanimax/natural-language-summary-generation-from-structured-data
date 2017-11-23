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
import keras


class MyLayer_fcn_bias(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer_fcn_bias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight and bias variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',shape = (1,self.output_dim),initializer='uniform',trainable=True)
        super(MyLayer_fcn_bias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        mul = K.dot(x, self.kernel) + self.bias
        return mul

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
