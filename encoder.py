from layers import MultiHeadSelfAttention, FeedFwd
from tensorflow.keras.layers import LayerNormalization
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,n_heads,d_model,units):
        """
        BERT Encoder layer class. The layer consist of the 
        feed forward and multihead self atention layer 

        :param n_heads: num of attention heads
        :param d_model: embedding dimension
        :param units: feed forward units
        """
        self.d = d_model
        self.nh = n_heads
        self.units = units
        super().__init__()

    def build(self,input_shape):
        self.mhsa = MultiHeadSelfAttention(self.d,self.nh)
        self.ffwd = FeedFwd(self.d,self.units)
        self.norm1 = LayerNormalization()
        self.norm2 = LayerNormalization()
        super().build(input_shape)

    def call(self,x):
        attn = self.mhsa(x)
        l = self.norm1(attn + x[0])
        ffwd = self.ffwd(l)
        return self.norm2(l + ffwd)




