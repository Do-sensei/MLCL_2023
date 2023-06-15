# 2023-1 MLCL 

## data analysis

### 1. Kaggle Dataset

#### Sign up [here](https://www.kaggle.com/)

1. Move `kaggle.json` in your desktop to `~/.kaggle/` in SSH

    - `scp -P "Port" "Path" "ID"@"IP":/home/"USER"/.kaggle/`
    - `cd ~`
    - `ls -a`
    - `cd .kaggle`
    - `ls`

#### Download Dataset
1. Korea Natural Gas Sales with Temperature [Kaggle](https://www.kaggle.com/datasets/zxtzxt30/korea-monthly-gas-sales-with-temperature)


2. Headgear 20 classes-Image Classification [Kaggle](https://www.kaggle.com/datasets/gpiosenka/headgear-image-classification)

3. 🏠📝 Airbnb Reviews: Wanderers' Delight & Stays!✨   [Kaggle](https://www.kaggle.com/datasets/omarsobhy14/airbnbreviews)


### 2. Prepare Dataset

#### Directory Structure

- `mkdir data`
- `cd data`
- `mkdir k-gas`
- `mkdir headgear`
- `mkdir airb`

#### Unzip Dataset

- `unzip k-gas.zip -d k-gas`
- `unzip headgear.zip -d headgear`
- `unzip airb.zip -d airb`

#### Check Dataset

- `cd k-gas`
- `ls`

#### Rename csv file

- `mv headgear.csv.csv headgear.csv`

### 3. Make your environment

#### Tmux

- `tmux new -s mlcl`
- `tmux a -t mlcl`

#### Anaconda

- `conda create -n mlcl python=3.8`
- `conda activate mlcl`

#### Install Packages

- `conda install Ipykernel`
- `conda install "Package Name"`
- `pip3 install "Package Name"`

### 4. Data Analysis

- `git clone https://github.com/Do-sensei/MLCL_2023.git`

#### 1. Korea Natural Gas Sales with Temperature

- [Korea Natural Gas Sales with Temperature](data_analysis_k-gas.ipynb)

- [Headgear 20 classes-Image Classification](data_analysis_headgear.ipynb)

- [🏠📝 Airbnb Reviews: Wanderers' Delight & Stays!✨]
    - To be updated

#### 2. Assginment

- [Assignment](1_Assignment.ipynb)

- End of the assignment: 2023.06.22 23:59

- Upload your assignment to **your github repository**