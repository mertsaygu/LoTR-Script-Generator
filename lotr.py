import os, dataprep, train, test,time
# import numpy as np 
# import tensorflow as tf 
# from tensorflow.keras.preprocessing.text import Tokenizer

# def split_input_target(chunk):
#     input_text = chunk[:-1]
#     output_Text = chunk[1:]
#     return input_text, output_Text


# base = os.getcwd()
# param_path = os.path.join(base,'parameters.json')
# data_path = os.path.join(base, 'data')
# file_path = os.path.join(data_path,'dataset.txt')
# checkpoint_dir = os.path.join(base,'training_checkpoints')

# seq_length, BATCH_SIZE, BUFFER_SIZE, EMBEDDING_DIM, RNN_UNITS, EPOCHS = dataprep.loadHyperParameters(param_path)

# data = dataprep.readTxt(file_path)
# vocab = sorted(set(data))
# char2idx = {u:i for i,u in enumerate(vocab)}
# idx2char = np.array(vocab)
# text_as_int = np.array([char2idx[c] for c in data])

# # examples_per_epoch = len(data)
# char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# sequences = char_dataset.batch(seq_length+1,drop_remainder = True)
# dataset = sequences.map(split_input_target)
# dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
# vocab_size = len(vocab)
# print(vocab_size)

# model = train.build_model(vocab_size=vocab_size,
#                     embedding_dim=EMBEDDING_DIM,
#                     rnn_units=RNN_UNITS,
#                     batch_size = BATCH_SIZE)



# # # for input_example_batch, target_example_batch in dataset.take(1):
# # #   example_batch_predictions = model(input_example_batch)
  
# # # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
# # # sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

# # #example_batch_loss  = loss(target_example_batch, example_batch_predictions)
# checkpoint_prefix = os.path.join(checkpoint_dir,'ckpt_{epoch}')
# checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath = checkpoint_prefix,
#     save_weights_only = True
# )
# start = time.time()
# history = train.train_model(model = model,dataset = dataset,epochs=100,callback=checkpoint_callback)
# print(f"training time: {time.time()-start}")
# # new_model = train.loadModelFromCheckpoint(model = model, path=checkpoint_dir)
# #print(new_model.summary())
# # print(test.generate_text(model=new_model,start_string=u"FRODO: "))
# test.plot_graphs(history,"accuracy")
# test.plot_graphs(history,"loss")
#print(test.generate_text(start_string="FRODO: "))
#model = train.load_model("model_final2.h5")
# base = os.getcwd()
# param_path = os.path.join(base,'parameters.json')
# data_path = os.path.join(base, 'data')
# file_path = os.path.join(data_path,'dataset.txt')
# checkpoint_dir = os.path.join(base,'training_checkpoints')
# saved_folder = os.path.join(base,"saved_models")

# seq_length, BATCH_SIZE, BUFFER_SIZE, EMBEDDING_DIM, RNN_UNITS, EPOCHS = dataprep.loadHyperParameters(param_path)
# data = dataprep.readTxt(file_path)
# vocab = sorted(set(data))
# vocab_size = len(vocab)
# model = train.build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
# model.summary()
#print(test.generate_text(model, "GOLLUM: ",num_to_generate=2000))

with json