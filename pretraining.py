from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling1D, Dense
import tensorflow as tf
import numpy as np
from tqdm import tqdm


class BertMLM(tf.keras.Model):
    def __init__(self, bert_model, mlm_units):
        """
        Masked Language model to pretrain BERT for MLM task
        :param bert_model: bert model (keras model class)
        :param mlm_units: vocabulary size (int)
        """
        super(BertMLM, self).__init__()
        self.bert = bert_model
        self.units = mlm_units
        self.batch_loss = {'train':[],'val':[]}
        self.epoch_loss = {'train':[],'val':[]}
        self.loss = SparseCategoricalCrossentropy()
        self.opt = Adam()

    def build(self,input_shape):
        self.mlm_layer = Dense(self.units, activation='softmax',
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))
        super().build(input_shape)

    def call(self,x):
        bert_out = self.bert(x)
        mlm_out = self.mlm_layer(bert_out)
        return mlm_out

    @tf.function
    def _train_step(self,x,y):
        with tf.GradientTape() as tape:
            pred = self(x,training=True)
            loss = self.loss(y,pred,sample_weight=y>0)
        weights = self.trainable_weights
        grads = tape.gradient(loss,weights)
        self.opt.apply_gradients(zip(grads,weights))
        return loss

    @tf.function
    def _test_step(self,x,y):
        pred = self(x,training=False)
        loss = self.loss(y,pred,sample_weight=y>0)
        return loss

    def train(self,train,val,epochs,lr_scheduler,sample_x,sample_y,vocab):
        for e in range(1,epochs+1):
            print(f'epochs : {e}/{epochs}')

            loss = 0
            for i,(x,s,y) in enumerate(tqdm(train)):
                lr = lr_scheduler(i+1)
                self.opt.learning_rate.assign(lr)
                x = (x,s)
                batch_loss = self._train_step(x,y)
                loss += batch_loss
                self.batch_loss['train'].append(batch_loss)

            loss = loss.numpy()/i
            self.epoch_loss['train'].append(loss)
            print('train loss :',round(loss,5))

            loss = 0
            for i, (x, s, y) in enumerate(val):
                x = (x, s)
                batch_loss = self._test_step(x, y)
                loss += batch_loss
                self.batch_loss['val'].append(batch_loss)

            loss = loss.numpy()/i
            print('val loss :',round(loss,5))

            if e == 1:
                self.save_weights('best_mlm.h5')
                print('weights saved')
            elif loss < min(self.epoch_loss['val']):
                self.save_weights('best_mlm.h5')
                print('weights saved')

            self.epoch_loss['val'].append(loss)
            print()
            self.predict_masked_tokens(sample_x,sample_y,vocab)
            print()

    def predict_masked_tokens(self,x,y,vocab):
        masked_idx = np.where(y > 0)[0]
        pred = self(x,training=False).numpy()[0]
        pred = np.argmax(pred,axis=-1)
        print('predicted tokens :', ' '.join(vocab.i2w[t] for t in pred[masked_idx]))
        print('actual tokens :', ' '.join(vocab.i2w[t] for t in y[masked_idx]))

    def summary(self,inputs):
        outputs = self.call(inputs)
        tf.keras.Model(inputs,outputs).summary()


class BertNSP(tf.keras.Model):
    def __init__(self,bert_model):
        """
        The model predicts if the 1st and 2nd sentence are continous
        :param bert_model: bert model (keras model class)
        """
        super(BertNSP,self).__init__()
        self.bert = bert_model
        self.batch_loss = {'train': [], 'val': []}
        self.epoch_loss = {'train': [], 'val': []}
        self.loss = BinaryCrossentropy()
        self.opt = Adam()

    def build(self,input_shape):
        self.nsp_layer = Dense(1,activation='sigmoid')
        self.pooling_layer = GlobalAveragePooling1D()
        super().build(input_shape)

    def call(self,x):
        bert_out = self.bert(x)
        pooling = self.pooling_layer(bert_out)
        nsp_out = self.nsp_layer(pooling)
        return nsp_out

    @ tf.function
    def _train_step(self,x,y):
        with tf.GradientTape() as tape:
            pred = self(x,training=True)
            loss = self.loss(y,pred)
        weights = self.trainable_weights
        grads = tape.gradient(loss,weights)
        self.opt.apply_gradients(zip(grads,weights))
        return loss

    @tf.function
    def _test_step(self, x, y):
        pred = self(x, training=False)
        loss = self.loss(y, pred)
        return loss

    def train(self,train,val,epochs,lr_scheduler):
        for e in range(1,epochs+1):
            print(f'epochs : {e}/{epochs}')

            loss = 0
            for i,(x,s,y) in enumerate(tqdm(train)):
                lr = lr_scheduler(i+1)
                self.opt.learning_rate.assign(lr)
                x = (x,s)
                batch_loss = self._train_step(x,y)
                loss += batch_loss
                self.batch_loss['train'].append(batch_loss)

            loss = loss.numpy()/i
            self.epoch_loss['train'].append(loss)
            print('train loss :',round(loss,4))

            loss = 0
            for i, (x, s, y) in enumerate(val):
                x = (x, s)
                batch_loss = self._test_step(x, y)
                loss += batch_loss
                self.batch_loss['val'].append(batch_loss)

            loss = loss.numpy()/i
            print('val loss :',round(loss,4))

            if e == 1:
                self.save_weights('best_nsp.h5')
                print('weights saved')
            elif loss < min(self.epoch_loss['val']):
                self.save_weights('best_nsp.h5')
                print('weights saved')

            self.epoch_loss['val'].append(loss)
            print()

    def summary(self,inputs):
        outputs = self.call(inputs)
        tf.keras.Model(inputs,outputs).summary()













        






