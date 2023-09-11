from layers import BERTEmbedding
from encoder import EncoderLayer
import tensorflow as tf


class BERT(tf.keras.Model):
    def __init__(self,d_model,vocab_size,pos_weights,
                 n_heads,units,n_encoder_layer=4, n_seg=2):
        """
        BERT model class
        :param d_model: embedding dimension
        :param vocab_size: vocab size
        :param pos_weights: positional encodings
        :param n_heads: num of heads
        :param units: feed forward units
        :param n_encoder_layer: num of encoder layers
        :param n_seg: num of segments
        """
        super(BERT,self).__init__()

        self.d = d_model
        self.vocab = vocab_size
        self.ne = n_encoder_layer
        self.pos_weights = pos_weights
        self.units = units
        self.maxlen = pos_weights.shape[0]
        self.nh = n_heads
        self.ns = n_seg

    def build(self,input_shape):
        self.emb = BERTEmbedding(self.d,self.vocab,self.pos_weights,self.ns)
        self.encoder_layers = [EncoderLayer(self.nh,self.d,self.units)]*self.ne
        super().build(input_shape)

    def call(self,x):
        x,mask = self.emb(x)

        for i in range(self.ne):
            x = self.encoder_layers[i]([x,mask])

        return x

    def summary(self,inputs):
        input_shape = [i.shape for i in inputs]
        self.build(input_shape)
        out = self.call(inputs)
        tf.keras.Model(inputs,out).summary()


