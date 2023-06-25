# [Assignment 3](./Assginment/) Gas Sales Forecasting with PyTorch

This repository contains code for a project that uses a Multilayer Perceptron (MLP) model implemented in PyTorch to forecast gas sales based on historical data.

## Project Structure

- `./configs/configs.yaml`: This file contains the configuration for the project, including the paths to the data and the model, the training parameters, and the model parameters.
    - *It needs to be modified* to run the code.

- `./utils/dataset.py`: This script defines the `GasDataset` class for data loading and preprocessing.

- `./utils/model_mlp.py`: This script defines the `MLP` class for the model.

- `training.py`: This script trains the model and saves it.

- `test.py`: This script loads the model and tests it.

## How to Run

1. Clone this repository.

2. Install the required packages.

3. Activate the conda environment and attach to the tmux session:

    - `~$ tmux attach -t mlcl`
    - `~$ conda activate mlcl`


4. Run `training.py` to train the model. The trained model will be saved to the path specified in `./configs/configs.yaml`.

5. Run `test.py` to test the model. The script will load the model from the path specified in `./configs/configs.yaml` and evaluate it on the test data.

## Data

The data used in this project is stored in a CSV file and includes the following columns:

- `Year`: The year of the data.
- `Month`: The month of the data.
- `Temperature`: The average temperature in that month.
- `Gangwondo`, `Seoul`, `Gyeonggido`, `Incheon`, `Gyeongsangnamdo`, `Gyeongsangbukdo`, `Gwangju`, `Daegu`, `Daejeon`, `Busan`, `Sejong`, `Ulsan`, `Jeollanamdo`, `Jeollabukdo`, `Jeju`, `Chungcheongnamdo`, `Chungcheongbukdo`: The gas sales in these regions.

## Model

The model used in this project is a Multilayer Perceptron (MLP) with one hidden layer. The model is implemented in PyTorch.

- If you have a time, Modify the model to improve performance.

## Training

The model is trained using the Mean Squared Error (MSE) loss and the Adam optimizer. The training script also includes a validation phase to monitor the model's performance on the validation data during training. After training, the model's parameters are saved to a file for later use.

- If you have a time, Modify the training code to improve performance.

## Testing

The testing script loads the trained model parameters from a file and evaluates the model's performance on the test data.
