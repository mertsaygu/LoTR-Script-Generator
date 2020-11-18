# LoTR Script Generator

In this project, I created a Neural Network that generates Lord of the Rings scripts

## Fastest way to start

- The fastest way to start is using pretrained model. All u need to do is using load_model funtion with the model's name as parameter.
"test.generate_text(train.load_model("model_final2.h5"), "GOLLUM: ",num_to_generate=2000)" 
This single line of code can generate 2000 characters long Lord of the Rings script that starts with "GOLLUM: "

## Script.py
"script.py" is the script version of the whole project. When you run the file, script creates the dataset, and the model, trains the model and generates 1000 character long Lord of the Rings script. It is not dynamic so it does not fetch the parameters from "parameters.json". It is completely free from all other files in the project. Written for test purposes.