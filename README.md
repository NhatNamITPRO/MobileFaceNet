
# MobileFaceNet

Face Recognition with MobileFaceNet model

## Overview

This repository contains the implementation of the MobileFaceNet model for face recognition. MobileFaceNet is a lightweight neural network designed for efficient face recognition on mobile and embedded devices.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/NhatNamITPRO/MobileFaceNet.git
cd MobileFaceNet
pip install -r requirements.txt
```

## Usage

The `main.py` script allows you to run different modes of the program, including preprocessing, training, and inference.

### Running Inference

#### Inference on an Image
To perform face recognition using the pre-trained MobileFaceNet model:

```bash
python main.py --mode inferal --path path/to/image.jpg
```

#### Inference on a Video
To perform inference on a video:

```bash
python main.py --mode inferal_video --path path/to/video.mp4
```

Demo results can be found in the `assets` folder.  
Example outputs:

![image](https://github.com/user-attachments/assets/6d5ab82a-627c-478d-8c9a-95d7fd0840e4)  
![image](https://github.com/user-attachments/assets/7ce0445c-629c-476f-b6de-2e8e48f7447c)

### Preprocessing

To preprocess the dataset, run:

```bash
python main.py --mode process
```

### Training

To train the MobileFaceNet model, run:

```bash
python main.py --mode train
```

### Configuration

The configuration for the project is specified in the `config.yml` file. Below is an example configuration:

```yaml
RAW_ROOT: 'dataset/raw'
PROCESSED_ROOT: 'dataset/processed/processed'
NOT_PROCESSED_ROOT: 'dataset/processed/not_processed'
BATCH_SIZE: 256
EPOCHS: 100
CHECK_POINT: null
BEST_MODEL: 'checkpoints/best_cpkt.ckpt'
SAVE_FREQ: 1
TEST_FREQ: 1
H: 112
W: 112
```

Descriptions:

- `RAW_ROOT`: Path to the folder containing raw, unprocessed data.  
  Example:

  ![image](https://github.com/user-attachments/assets/823620fe-5574-4925-a5f5-e8109d7b7779)  
  ![image](https://github.com/user-attachments/assets/7ec4d729-d7f5-439a-b1c1-5766e32429db)

- `PROCESSED_ROOT`: Path to the folder containing processed data.  
  Example:

  ![image](https://github.com/user-attachments/assets/27cbaa1a-d90f-4e7d-affc-7731b27d9c15)  
  ![image](https://github.com/user-attachments/assets/b18825c0-6346-4c7e-b2e4-234438af45a1)

- `NOT_PROCESSED_ROOT`: Path to the folder for data that failed processing.  
  Example:

  ![image](https://github.com/user-attachments/assets/b2f0f7e7-d3f1-4b05-83b3-dd413a08385e)

---

## Datasets

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition). Make sure to organize the data according to the configuration described above.
