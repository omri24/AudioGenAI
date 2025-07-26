# 🎯 AudioGenAI

The process of composing music contains many challenges - from timing, choosing notes and coordination between musical instruments. The goal of this project is to help composers with this challenges. 
The motivation of this project is to create an assistant that helps composers and music producers. This project aims to develop a system that can interpret corrupted MIDI inputs and generate corrected versions.

---

## 🚀 How to Train

For training use this notebook project_DL_training_notebook.ipynb :
### 🔧 Step 1: Setup
install the required libraries, mount google drive and clone git repository.
### 📁 Step 2: Configure Paths
Specify: full project path in google drive and name of the folder to save training parameters
### 📦 Step 3: Create and or  Dataset
-If need to create dataset:
  use create_dataset function in MIDI_DATASET - takes folder path of multiple midi format songs, adds errors and divides songs into the length of CNN input. Saves dataset in pth format.
-Load Dataset - using load_dataset class from MIDI_DATASET file.
-Use the split_train_validation function to create 85% train and 15% validation. 
### ⚙️ Step 4: Select Training Device
Set device for training (cpu or gpu)
### 🤖 Step 5: Train Models
When running the training notebook, trains eight different models for compare:
Two architectures:
    1- CNN
    2 - Transformers
For each model four types of tests:
    1 - regular cross entropy + weighted penalty based on the distance between predicted and true classes
    2 - regular cross entropy + penalty based on the distance between the prediction and the previous target
    3 - using regual cross entropy and both weighted penalties
    4 - only using regular cross entropy
    
To train each model, use the `train_and_test` class from the `DL_TRAIN_TEST` file:
a. define model type - CNN or Transformers
b. training and validation datasets for training
c. path to save model
The set_model function in class then sets the model parameters, depending on model type.
The set_loss_function in class sets up loss function as mentioned above.
The train function trains the model depending on the number of epochs entered.
The save_model_weights saves the trained weights in the path defined above
The save_training function in class saves the training results (loss function improvements over epochs)
### 📊 Step 6: Compare Results
Run the last block in notebook "Compare all the training results in one plot." Which takes all the training results from all eight different model training and gives a graph for comparission. 
   
## 🚀 How to Load and Test
