from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Activation, concatenate
from keras.optimizers import Adam
import numpy as np

import utils

class NER_Model():
    
    def __init__(self, num_tokens, indexDicts, batch_size, hidden_size, learning_rate, num_features):
        self.model = None
        self.num_tokens = num_tokens
        self.word2index = indexDicts['word2index']
        self.pos2index = indexDicts['pos2index']
        self.entpos2index = indexDicts['entpos2index']
        self.tag2index = indexDicts['tag2index']
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.num_features = num_features
        
        self.build()
        
    def build(self):
        in1 = Input(shape=(self.num_tokens, ))
        feed = Embedding(len(self.word2index), self.batch_size)(in1)
        inputs = [in1]
        if self.num_features > 1:
            # num_features == 2 is for POS data
            in2 = Input(shape=(self.num_tokens, ))
            emb2 = Embedding(len(self.pos2index), self.batch_size)(in2)
            feed = concatenate([feed, emb2])
            inputs.append(in2)
        if self.num_features > 2:
            # num_features == 3 is for Entity-POS data
            in3 = Input(shape=(self.num_tokens, ))
            emb3 = Embedding(len(self.entpos2index), self.batch_size)(in3)
            feed = concatenate([feed, emb3])
            inputs.append(in3)
        bid_lstm = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(feed)
        timeDist = TimeDistributed(Dense(len(self.tag2index)))(bid_lstm)
        fin_out = Activation('softmax')(timeDist)
        self.model = Model(inputs=inputs, outputs = fin_out)

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(self.learning_rate),
                      metrics=['accuracy'])

        print (self.model.summary())
        
    def fit(self, inputs, train_tags_y, epochs):
        if self.num_features != len(inputs):
            print ("ERROR: Number of features and inputs don't match. Check num_features parameter.")
        self.model.fit(inputs, 
                       utils.to_categorical(
                        train_tags_y, len(self.tag2index)), 
                       batch_size=self.batch_size, 
                        epochs=epochs, 
                       validation_split=0.2, 
                       verbose=2)

    def predict_class(self, inputs):
        if self.num_features != len(inputs):
            print ("ERROR: Number of features and inputs don't match. Check num_features parameter.")
        outputs = self.model.predict(inputs)
        predictions = np.argmax(outputs, 2)
        return predictions