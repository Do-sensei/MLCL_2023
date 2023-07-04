# **[Assignment 5]** HeadGear Image Generation with PyTorch using DCGAN

## ⛑️ HeadGear Image Generation with PyTorch using DCGAN 🤖

Welcome to Assignment 5. In this assignment, you will be using Deep Convolutional Generative Adversarial Networks (DCGAN) to generate images of headgears. This is an exciting opportunity to apply your skills in AI and Machine Learning in a creative and challenging way.

- ⏰ *Due Date: 2023.07.04(TUE) 23:59*

## Project Structure

The project has the following structure:

- [`./configs/`](./configs/): This directory contains the configuration files for the model.

- [`./model/`](./model/): This directory will contain your implemented DCGAN model.

- [`./utils/`](./utils/): This directory contains utility scripts including [`dataset.py`](./utils/dataset.py) for data loading and preprocessing, and [`DCGAN.py`](./utils/DCGAN.py) for the DCGAN architecture.

- [`training.py`](training.py): This script will use the data and the DCGAN model to train.

- [`test.ipynb`](test.ipynb): This Jupyter notebook will be used for testing the trained model and visualizing the results.

## How to Run

1. Clone this repository.

2. Activate the conda environment and attach to the tmux session:

    - `~$ tmux attach -t mlcl`
    - `~$ conda activate mlcl`

3. Install the required packages.

4. Modify the [`configs.yaml`](./configs/configs.yaml) file according to your paths for training, validation, and test datasets, as well as model saving path and training parameters.

5. Run `~$ python training.py` to train the model. The trained model will be saved to the path specified in [`configs.yaml`](./configs/configs.yaml).

6. After training, open the Jupyter notebook [`test.ipynb`](./test.ipynb) to test the model and visualize the generated images.

## Data

The dataset for this project consists of various types of headgear images, categorized into classes such as ASCOT CAP, BASEBALL CAP, TOP HAT, ZUCCHETTO, and many more. The dataset is stored in a CSV file with the following columns:

- `class id`: The ID corresponding to the class of the headgear.
- `filepaths`: The path to the image file.
- `labels`: The name of the class (i.e., the type of headgear).
- `data set`: The subset of the dataset to which the image belongs (i.e., train, test, or validation).

The dataset is divided into three subsets:

- `Train set`: This set includes images used for training the DCGAN model.
- `Test set`: This set includes images used for evaluating the performance of the trained DCGAN model.
- `Valid set`: This set includes images used for validating the model's performance during training.

The `HeadGearDataset` class in [`./utils/dataset.py`](./utils/dataset.py) script takes care of loading and preprocessing the dataset. It reads the CSV file, loads the images from the provided filepaths, and assigns the corresponding labels and class IDs. It's crucial to properly configure the dataset path and preprocessing steps in the `HeadGearDataset` class based on your dataset structure and requirements.

Do remember, in DCGAN, we are interested in the distribution of different types of headgear images. While the labels or class ids may not directly influence the training of the DCGAN, they might be helpful for you to assess or visualize the performance of your model.

Note: The paths and structure of the dataset should be properly set up to align with the dataset used in this project.

## Model

You will be using the DCGAN model for this task. The architecture for the model is defined in [`./utils/DCGAN.py`](./utils/DCGAN.py). Please understand the architecture and the working of DCGAN before starting the training process.

## Training

The training process involves running the [`training.py`](./training.py) script. This script will use the configurations from [`./configs/configs.yaml`](./configs/configs.yaml), data processed by [`./utils/dataset.py`](./utils/dataset.py), and the DCGAN model from [`./utils/DCGAN.py`](./utils/DCGAN.py) to train. The trained model will be saved in the [`./model/`](./model/) directory.

## Testing

After training the model, you can test it using the [`test.ipynb`](./test.ipynb) Jupyter notebook. This notebook will load the trained model, generate images of headgears, and provide utilities for visualizing these generated images.

Remember to observe the generated images and compare them with the training data to get a sense of how well your model has learned to generate new images.

Good luck with your assignment!