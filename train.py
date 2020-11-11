import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, LSTM, Bidirectional
import os, time, dataprep
from datetime import datetime



base = os.getcwd()
base = os.path.join(base,"LoTR-Script-Generator")
param_path = os.path.join(base,'parameters.json')
data_path = os.path.join(base, 'data')
file_path = os.path.join(data_path,'dataset.txt')
checkpoint_dir = os.path.join(base,'training_checkpoints')
saved_folder = os.path.join(base,"saved_models")

param = dataprep.loadHyperParameters(param_path)
data = dataprep.readTxt(file_path)
vocab = sorted(set(data))
vocab_size = len(vocab)

class mycallBack(tf.keras.callbacks.Callback):
    '''
    Callback to stop training at desired accuracy rate which is 0.98 
    '''
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get("accuracy")>0.98):
            self.model.stop_training = True

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    '''
    Creates a sequential model, compile it and return it.
    Parameters:
    ----------
    
    vocab_size: int
        Number of different characters in dataset
    embedding_dim: int 
        Hyperparameter used as number embedding dimensions, read from parameters.json
    rnn_units: int 
        Hyperparameter used as number of RNN units in a single layer, read from parameters.json
    batch_size : int 
        Hyperparameter used as the size of batch, read from parameters.json
        
    Output: 
    ----------
    
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    embedding (Embedding)        (64, None, EMBEDDING_DIM)           16640
    _________________________________________________________________
    gru (GRU)                    (64, None, RNN_UNITS)          3938304
    _________________________________________________________________
    gru_1 (GRU)                  (64, None, RNN_UNITS)          6297600
    _________________________________________________________________
    dense (Dense)                (64, None, 256)           262400
    _________________________________________________________________
    dense_1 (Dense)              (64, None, 256)           65792
    _________________________________________________________________
    dense_2 (Dense)              (64, None, VOCAB_SIZE+1)            16705
    =================================================================
    Total params: 10,597,441
    Trainable params: 10,597,441
    Non-trainable params: 0
    _________________________________________________________________
    
    '''
    model = Sequential([
        Embedding(vocab_size,
                  embedding_dim,
                  batch_input_shape = [batch_size,None]),
        GRU(rnn_units,
            return_sequences = True,
            stateful = True,
            recurrent_initializer = 'glorot_uniform'),
        GRU(rnn_units,
            return_sequences = True,
            stateful = True,
            recurrent_initializer = 'glorot_uniform'),
        Dense(256),
        Dense(256),
        Dense(vocab_size)
    ])
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",metrics=['accuracy'])
    return model


def train_model(dataset, epochs):
    '''
    Trains the model, Saves it to the folder and returns the history
    
    Parameters
    -------------
    dataset : str
        dataset to fed through model
    epochs: int
        number of epochs
    
    Output: tensorflow.keras.model.history
        History of training 
    '''
    model = build_model(vocab_size, param["embedding_dim"], param["rnn_units"], param["batch_size"])
    checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt_{epoch}')
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = checkpoint_prefix,
        save_weights_only = True
    )
    history = model.fit(dataset, epochs = epochs, callbacks = [checkpoint_callback, mycallBack()])
    save_model(model)
    return history



def load_model(model_name):
    '''
    Loads model from folder with a given name
    '''
    model_path = os.path.join(saved_folder,model_name)
    model = tf.keras.models.load_model(model_path)
    # model.build(tf.TensorShape([1,None]))
    model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    return model
    

def loadModelFromCheckpoint(model,path):
    '''
    Loads latest checkpoint to a given model
    
    Parameters
    ----------
    path: str   
        Path of the folder of checkpoints
    '''
    model.load_weights(tf.train.latest_checkpoint(path))
    model.build(tf.TensorShape([1, None]))
    return model

def save_model(model):
    '''
    Saves given model with a time stamp. 
    '''
    ts = (str)(datetime.now()).split(".")
    ts = ts[0].replace(" ","_").replace(":","'")
    model_name = "model_"+ts+".h5"
    model_path = os.path.join(saved_folder,model_name)
    model.save(model_path)