# 🎯 AudioGenAI

The process of composing music contains many challenges - from timing, choosing notes and coordination between musical instruments. The goal of this project is to help composers with this challenges. 
The motivation of this project is to create an assistant that helps composers and music producers. This project aims to develop a system that can interpret corrupted MIDI inputs and generate corrected versions.

---

## 🚀 How to Train

For training use this notebook **project_DL_training_notebook.ipynb** :
### 🔧 Step 1: Setup
install the required libraries, mount google drive and clone git repository.
### 📁 Step 2: Configure Paths
Specify: full project path in google drive and name of the folder to save training parameters
```python
project_path = ""
folder_name = ""
```
### 📦 Step 3: Create and Load Dataset
-If need to create dataset:

  use create_dataset function in MIDI_DATASET - takes folder path of multiple midi format songs, adds errors and divides songs into the length of CNN input. Saves dataset in pth format.

For training used maestro dataset V3 - https://magenta.withgoogle.com/datasets/maestro 
  
```python
ds_train = MIDI_DATASET.create_dataset(path="8", length=64, output_file="maestro_dataset_for_train_and_validation")
```

-Load Dataset - using load_dataset class from MIDI_DATASET file.
-Use the split_train_validation function to create 85% train and 15% validation. 

```python
import MIDI_DATASET
dataset = MIDI_DATASET.load_dataset(f"{project_path}/dataset/maestro_dataset_for_train_and_validation.pth")
train_ds, validation_ds = dataset.split_train_validation(0.85, 0.15, seed=42)
```

### ⚙️ Step 4: Select Training Device
Set device for training (cpu or gpu)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

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
Run the last block in notebook "Compare all the training results in one plot." Which takes all the training results from all eight different model training and generate a graph for comparission. 
   
## 🧪 How to Load and Test

You can use the trained model weights from this Google Drive folder:  
🔗 [Pretrained Models](https://drive.google.com/drive/folders/1i65Cp7_PfqBQ-K-DWws-QbCQJE0U7zlX)

---

### 📦 Step 1: Load Test Dataset

Use the `load_test_dataset` function from the `MIDI_DATASET` file:

```python
from MIDI_DATASET import load_test_dataset
test_dataset = load_test_dataset("dataset/maestro_dataset_for_test.pth")
```

---

### 🧠 Step 2: Load Model and Run Tests

For each model to be tested:

- Create a `Test` class instance from the `DL_TRAIN_TEST` file.  
  It takes:
  - `device`: either `"cpu"` or `"gpu"`
  - `path`: path to the trained model (without the `.pth` extension)
 
```python
test_type = [train_and_test.MODEL_TYPE.CNN, train_and_test.MODEL_TYPE.TRANSFORMER]
test_names = ["_weights_target_0_weight_before_0_cross_entropy","_weights_target_0_weight_before_1_cross_entropy","_weights_target_1_weight_before_0_cross_entropy", "_weights_target_1_weight_before_1_cross_entropy"]
for t in test_type:
  for i, tests in enumerate(test_names):
    test  = train_and_test.Test(device=device, path =fr"{output_path}\{t}{tests}")
```
#### 🔄 Load Model

Use the `load_model()` function inside the `Test` class:

```python
    test.load_model(model_type=t, sequence_len=64, hidden_dim = 512, encoder_layers = 4, decoder_layers=2)
```

- Loads both trained and untrained versions (for IM) of the model for testing.
- Trained weights are loaded from `self.path + ".pth"`.

#### 🧪 Run Model on Test Set

Use `test_multiple_from_dataset()` to run the test:

```python
test_instance.test_multiple_from_dataset(
    test_dataset=test_dataset,
    output_path=output_path",
    test_name=f"{t}{tests}"
)
```

This function will:
- Run inference on the test dataset.
- Save 3 output songs during the process for analysis.
- Log test results for each run.

---

### 💾 Step 3: Save Test Results

Use the `save_test_results()` function to export results to a CSV file:

```python
test_instance.save_test_results(results_file_path="results/test_results.csv", test_name=f"{t}{tests})
```

Saved results include the following metrics:

- `"test name"`
- `"cross_entropy_no_weights"`
- `"cross_entropy_weights_target"`
- `"cross_entropy_weights_prev"`
- `"im target L1 trained vs untrained model output"`
- `"im target L1 trained vs error vector"`
- `"im target harmonic vs error vector"`
- `"im prev target harmonic vs error vector"`
- `"im prev output harmonic vs error vector"`
- `"im target harmonic vs untrained model output"`
- `"im prev target harmonic vs untrained model output"`
- `"im prev output harmonic vs untrained model output"`
- `"im target warrestein trained vs untrained model output"`
- `"im target warrestein trained vs error vector"`
- `"inference time"`
  

