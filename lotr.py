import train, test

test.generate_text(train.load_model("model_final2.h5"), "GOLLUM: ",num_to_generate=2000)

