import tensorflow as tf 
import os, dataprep,train,test
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def split_input_target(chunk):
    '''
    Splits chunks to input and output text
    '''    
    input_text = chunk[:-1]
    output_Text = chunk[1:]
    return input_text, output_Text

base = os.getcwd()
param_path = os.path.join(base,'parameters.json')
data_path = os.path.join(base, 'data')
file_path = os.path.join(data_path,'dataset.txt')
checkpoint_dir = os.path.join(base,'training_checkpoints')
generated_dir = os.path.join(base,"generated")

param = dataprep.loadHyperParameters(param_path)

data = dataprep.readTxt(file_path)
vocab = sorted(set(data))
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in data])

examples_per_epoch = len(data)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1,drop_remainder = True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(param['buffer_size']).batch(param['batch_size'],drop_remainder=True)
vocab_size = len(vocab)

def generate_text(model,start_string,num_to_generate=1000,temperature = 1.0):
   '''
   Predictor
   
   Parameters
   ----------
   model : tf.keras.Sequential
        Trained model to make prediction
    start_string : str 
        Starting string to generate prediction 
    num_to_generate : int 
        Length of the generated text 
        
    Output 
    ----------
    Returns string
   '''
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval,0) 
    print(input_eval)
    text_generated = []
    model.reset_states()
    for i in range(num_to_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions,0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions,num_samples = 1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id],0)
        text_generated.append(idx2char[predicted_id])
    generatedText(start_string + ''.join(text_generated))
    return (start_string + ''.join(text_generated))

def plot_graphs(history):
    '''
    Plots Accuracy vs Loss
    '''
    plt.plot(history.history["accuracy"],"r")
    plt.plot(history.history["loss"],"b")
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend(["accuracy", "loss"])
    plt.show()

def generatedText(text):
    '''
    Saves generated text to a .txt file 
    '''
    ts = (str)(datetime.now()).split(".")
    ts = ts[0].replace(" ","_").replace(":","'")
    fileName = f"generated@{ts}.txt"
    fileDir = os.path.join(generated_dir,fileName)
    f = open(fileDir,"w+",encoding="utf-8")
    f.write(text)
    f.close()

