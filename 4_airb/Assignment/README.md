# **[Assignment 6]** Airbnb Review Classification with PyTorch

## 🏠📝 Airbnb Review Classification using BERT and PyTorch 🤖

Welcome to Assignment 6. In this assignment, you will apply your skills in AI and Deep Learning to classify Airbnb reviews. Specifically, you will leverage the BERT model (*'bert-base-multilingual-uncased-sentiment'*) from the [HuggingFace library](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment), implemented in PyTorch.

- ⏰ *Due Date: 2023.07.06(THU) 23:59*

## Project Structure

The structure of the project is as follows:

- [`./configs/configs.yaml`](./configs/configs.yaml): This file contains the configuration for the project, including the paths to the data and the model, the training parameters, and the model parameters.
- [`./model/`](./model/): This directory will store your trained BERT model.
- [`./utils/`](./utils/): This directory contains utility scripts, including [`dataset.py`](./utils/dataset.py) for data loading and preprocessing.
- [`training.py`](training.py): This script uses the data and BERT model for training.
- [`test.py`](./test.py): This script is used for testing the trained model and calculating performance metrics.

## How to Run

1. Clone this repository.
2. Activate the conda environment and attach to the tmux session:

    - `~$ tmux attach -t mlcl`
    - `~$ conda activate mlcl`

3. Install the required packages.
4. Update the [`./configs/configs.yaml`](./configs/configs.yaml) file with your specified paths for the datasets, model saving path, and training parameters.
5. Run `~$ python training.py` to train the model. The trained model will be saved to the path specified in [`./configs/configs.yaml`](./configs/configs.yaml).
6. After training, run `~$ python test.py` to evaluate the model and print the performance metrics.

## Data

The dataset for this project, 'AirBNBReviews.csv', contains Airbnb reviews classified into sentiment categories. The dataset includes the following columns:

- `Genre`: The aspect of the stay that the review addresses (e.g., location, cleanliness).
- `Review`: The actual text of the review.
- `Positive or Negative`: A binary value indicating whether the review is positive (**1**) or negative (**0**).

The dataset should be partitioned into training, validation, and test sets. The `ReviewDataset` class in the [`./utils/dataset.py`](./utils/dataset.py) script manages the loading and preprocessing of the dataset. It reads the CSV file, processes the reviews, and assigns the corresponding labels. Be sure to correctly configure the dataset path and preprocessing steps in the `ReviewDataset` class based on your dataset structure and needs.

## Model

You will use the BERT model (*'bert-base-multilingual-uncased-sentiment'*) from the [HuggingFace library](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) for this task. Make sure you are familiar with the BERT architecture and how it operates before beginning the training process.

## Training

To train the model, execute the [`training.py`](training.py) script. This script employs the configurations from [`./configs/configs.yaml`](./configs/configs.yaml), data processed by [`./utils/dataset.py`](./configs/configs.yaml), and the BERT model for training. The trained model will be saved in the [`./model/`](./model/) directory.

## Testing

After training the model, execute the [`test.py`](test.py) script to evaluate the model and print the performance metrics, which include accuracy, precision, recall, F1 score, and the confusion matrix.

Best of luck with your assignment!