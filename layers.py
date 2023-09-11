from tensorflow.keras.layers import Embedding,Dropout,Dense
import tensorflow as tf
from tensorflow import keras

l2_reg = keras.regularizers.l2(0.01)


class BERTEmbedding(tf.keras.layers.Layer):
    def __init__(self,d_model,vocab_size,pos_weights,n_seg=2):
        """
        BERT joint embeddings layer class
        :param d_model: size of embedding dimension
        :param vocab_size: vocabulary size
        :param pos_weights: positional encodings
        :param n_seg: num of segmnets
        """
        super().__init__()
        self.d = d_model
        self.vocab = vocab_size
        self.pos_weights = pos_weights
        self.maxlen = pos_weights.shape[0]
        self.ns = n_seg

    def build(self,input_shape):
        self.word_emb = Embedding(self.vocab,self.d)
        self.pos_emb = Embedding(self.maxlen,self.d,weights=[self.pos_weights],trainable=False)
        self.seg_emb = Embedding(self.ns, self.d)
        super().build(input_shape)

    def call(self, x):
        pos = tf.range(0, self.maxlen, 1)
        x,seg = x  # (tokenized sequence, segment ids)
        mask = x>0  # boolean attention mask
        x = self.word_emb(x) + self.pos_emb(pos) + self.seg_emb(seg)
        return x,mask


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self,d_model,n_heads):
        """
        Multihead attention layer class
        :param d_model: embedding dimension
        :param n_heads: num attention heads
        """
        self.d = d_model
        self.nh = n_heads
        assert d_model % n_heads == 0
        self.dh = d_model // n_heads
        self.attention_scores = None
        super().__init__()

    def build(self,input_shape):
        self.qw = Dense(self.d,kernel_regularizer=l2_reg)
        self.kw = Dense(self.d,kernel_regularizer=l2_reg)
        self.vw = Dense(self.d,kernel_regularizer=l2_reg)
        self.fc = Dense(self.d,kernel_regularizer=l2_reg)
        self.drop_probs = Dropout(0.1)
        super().build(input_shape)

    def split_heads(self, inputs, batch_size):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.reshape(
            inputs, shape=(batch_size, inputs.shape[1], self.nh, self.dh))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, x):

        x,mask = x

        # q -> query vector
        # k -> key vector
        # v -> value vector

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)
        b = x.shape[0]

        # create attention heads
        q = self.split_heads(q,b)
        k = self.split_heads(k,b)
        v = self.split_heads(v,b)

        # apply attention mask and get softmax attention score
        score = tf.matmul(q,k,transpose_b=True)/self.d**0.5
        mask = tf.expand_dims(tf.expand_dims(mask,1),1)
        score = tf.where(mask,score,-1e9)
        score = self.drop_probs(score)
        score = tf.nn.softmax(score, axis=-1)
        self.attention_scores = score

        # get attention vector
        attn = tf.matmul(score,v)
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn,(-1, tf.shape(attn)[1], self.d))
        attn = self.fc(attn)
        return attn


class FeedFwd(tf.keras.layers.Layer):
    def __init__(self,d_model,units):
        """
        Feed forward layer
        :param d_model: embedding dimension
        :param units: feed forward units
        """
        self.d = d_model
        self.units = units
        super().__init__()

    def build(self,input_shape):
        self.fc2 = Dense(self.d,kernel_regularizer=l2_reg)
        self.fc1 = Dense(self.units,activation='gelu',kernel_regularizer=l2_reg)
        self.drop = Dropout(0.1)
        super().build(input_shape)

    def call(self, x):
        o = self.fc1(x)
        o = self.fc2(o)
        o = self.drop(o)
        return o


