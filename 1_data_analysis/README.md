# 2023-1 MLCL 

## 📈 Data analysis 📊

### 1. Kaggle Dataset

#### Sign up [here](https://www.kaggle.com/)

1. Move `kaggle.json` in your desktop to `~/.kaggle/` in SSH

    - `$ scp -P "Port" "Path" "ID"@"IP":/home/"USER"/.kaggle/`
    - `$ cd ~`
    - `$ ls -a`
    - `$ cd .kaggle`
    - `$ ls`

2. Install Kaggle API

    - `pip3 install kaggle`

3. Clone this repository

    - `$ git clone https://github.com/Do-sensei/MLCL_2023.git`: Clone this repository in 'mlcl' directory


4. Generate Directory Structure For Kaggle Dataset

    - `~/MLCL_2023$ mkdir data`: Make 'data' directory
    - `~/MLCL_2023$ cd data`: Move to 'data' directory
    - `~/data$ mkdir k-gas`: Make 'k-gas' directory
    - `~/data$ mkdir headgear`: Make 'headgear' directory
    - `~/data$ mkdir airb`: Make 'airb' directory
#### Download Dataset
1. [Korea Natural Gas Sales with Temperature](https://www.kaggle.com/datasets/zxtzxt30/korea-monthly-gas-sales-with-temperature)

2. [Headgear 20 classes-Image Classification](https://www.kaggle.com/datasets/gpiosenka/headgear-image-classification)

3. [🏠📝 Airbnb Reviews: Wanderers' Delight & Stays!✨](https://www.kaggle.com/datasets/omarsobhy14/airbnbreviews)


### 2. Prepare Dataset

#### Unzip Dataset

- `~/data$ unzip k-gas.zip -d k-gas`: Unzip dataset to 'k-gas' directory
- `~/data$ unzip headgear.zip -d headgear`: Unzip dataset to 'headgear' directory
- `~/data $ unzip airb.zip -d airb`: Unzip dataset to 'airb' directory

#### Check Dataset

- `~/data$ cd k-gas`
- `~/data/k-gas$ ls`: Check dataset

#### Rename csv file

- `~/data/headgear$ mv headgear.csv.csv headgear.csv`

### 3. Make your environment

#### Tmux

- `$ tmux new -s mlcl`: Create new session
- `$ tmux a -t mlcl`: Attach to session

#### Anaconda

- `$ conda create -n mlcl python=3.8`: Create new environment with python 3.8
- `$ conda activate mlcl`: Activate environment

#### Install Packages

- `$ conda install Ipykernel`: Install Ipykernel(Jypyter kernel for python)
- `$ conda install "Package Name"`
- `$ pip3 install "Package Name"`

### 4. 📈 Data Analysis 

#### 1. Kaggle Dataset for class

- [Korea Natural Gas Sales with Temperature](data_analysis_k-gas.ipynb)

- [Headgear 20 classes-Image Classification](data_analysis_headgear.ipynb)

- [🏠📝 Airbnb Reviews: Wanderers' Delight & Stays!✨](data_analysis_airb.ipynb)
    - We will use the `Bert` Model from [Hugging Face](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)


#### 2. Assginment ✍️

- 📤 Upload your assignment to **your github repository** 

- ❗Caution❗: ***Do not upload Kaggle Dataset*** to your github repository

- 💻 In next class, we will check your assignment in your github repository
##### 1. Data Analysis

- ~~⏰ End of the assignment: *2023.06.21(WED) 23:59* (*Finished*)~~
    - ~~[k-gas Assignment](1_Assignment_k-gas.ipynb)~~

- ~~⏰ End of the assignment: *2023.06.25(SUN) 23:59* (*Finishied*)~~
    - ~~[headgear Assignment](1_Assignment_headgear.ipynb)~~
    - ~~[airb Assignment](1_Assignment_airb.ipynb)~~