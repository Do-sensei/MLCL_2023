# **[Assignment 4]** HeadGear Classification with PyTorch

## ⛑️ PyTorch for Headgear Classification 🤖

- ⏰ *Due Date: 2023.07.02(SUN) 23:59*

Welcome to HeadGear classification using PyTorch and ResNet-50!

This project classifies different types of headgear (hats, helmets, etc.) using PyTorch and ResNet-50. We use a custom ResNet-50 architecture with Bottleneck layers and Dropout. The project also involves image preprocessing and data augmentation.

## Project Structure

- [`./configs/configs.yaml`](./configs/configs.yaml): This file contains the configuration for the project, including the paths to the data and the model, the training parameters, and the model parameters. *Please modify this according to your paths*.

- [`./utils/dataset.py`](./utils/dataset.py): This script defines the `HeadGearDataset` class for data loading and preprocessing.

- [`./utils/resnet_50.py`](./utils/resnet_50.py): This script defines the `ResNet-50` class for the model.

- [`./utils/config.py`](./utils/config.py): This script is used to load the configurations from `configs.yaml`.

- [`training.py`](training.py): This script trains the model using the training data and validates it using the validation data. The trained model is then saved to the specified path.

- [`test.py`](test.py): This script loads the trained model and tests it using the test data.

## How to Run

1. Clone this repository.

2. Activate the conda environment and attach to the tmux session:

    - `~$ tmux attach -t mlcl`
    - `~$ conda activate mlcl`

3. Install the required packages.

4. Modify the [`./configs/configs.yaml`](./configs/configs.yaml) file according to your paths for training, validation, and test datasets, as well as model saving path and training parameters.

5. Run `~$ python3 training.py` to train the model. The trained model will be saved to the path specified in [`./configs/configs.yaml`](./configs/configs.yaml).

6. Run `~$ python3 test.py` to test the model on your test data. The test accuracy and F1 score will be printed on the console.

## Data

The dataset used in this project is stored in a CSV file and includes the following columns:

- `class id`: The ID corresponding to the class of the headgear.
- `filepaths`: The path to the image file.
- `labels`: The name of the class (i.e., the type of headgear).
- `data set`: The subset of the dataset to which the image belongs (i.e., train, test, or validation).

The dataset is split into three sets:
- `Train set`: Contains images used for training the model.
- `Test set`: Contains images used for testing the trained model's performance.
- `Valid set`: Contains images used for validation during model training.

The dataset contains various headgear categories such as ASCOT CAP, BASEBALL CAP, TOP HAT, ZUCCHETTO, etc.

The data is loaded using a custom data loader, which reads the CSV file, loads the images from the provided filepaths, and assigns the corresponding labels and class IDs.

The provided `HeadGearDataset` class in [`./utils/dataset.py`](./utils/dataset.py) is responsible for loading and preprocessing the dataset. Please make sure to configure the dataset path and preprocessing steps in the `HeadGearDataset` class according to your dataset structure and requirements.

Note: The paths and structure of the dataset should be properly configured and aligned with the dataset used in this project.


## Model

The model used in this project is a ResNet-50, which is a deep convolutional neural network architecture. It has been modified to classify headgear images into different classes. The model is implemented in PyTorch.

- If you have time, you can modify the model to further improve its performance.


## Training

The model is trained using the CrossEntropyLoss and the Adam optimizer. The training script includes a training loop that iterates over the dataset and performs forward and backward propagation to update the model's parameters. After each epoch, the model's performance is evaluated on the validation data to monitor its progress.

- If you have time, you can modify the training code to further improve its performance.


## Testing

The testing script loads the trained model parameters from a file and evaluates the model's performance on the test data. The accuracy and F1 score are calculated to measure the model's performance.

*Note: Before running the code, make sure to modify the necessary parts such as file paths and dataset configuration in the [./configs/configs.yaml](./configs/configs.yaml) file to match your setup.*