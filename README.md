---

# Chest X-Ray Pneumonia Detection Project

This repository contains the code and resources for detecting pneumonia from chest X-ray images using deep learning techniques. The primary objective of this project is to build a robust model that can identify pneumonia cases from X-ray images, assisting medical professionals in the diagnosis process.

## Table of Contents

- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

Pneumonia is a respiratory infection that causes inflammation in the air sacs in one or both lungs. This infection can range from mild to life-threatening. Timely and accurate detection of pneumonia is crucial for effective treatment. Radiologists typically use chest X-rays to diagnose pneumonia, but the process can be time-consuming and error-prone.

This project leverages convolutional neural networks (CNNs) to automate the pneumonia detection process using chest X-ray images. It aims to achieve high accuracy and reliability in identifying pneumonia cases from X-ray images.

## Project Overview

In this project, we employ a deep learning model to classify chest X-ray images into two categories: 
- **Normal**
- **Pneumonia**

The project consists of the following steps:
1. Data collection and preprocessing.
2. Building a Convolutional Neural Network (CNN) for image classification.
3. Training and evaluating the model on the dataset.
4. Saving and deploying the trained model for predictions.

## Dataset

The dataset used in this project is publicly available and contains X-ray images labeled as either **Normal** or **Pneumonia**. The dataset is organized into three folders: **train**, **test**, and **val** (validation). 

The dataset structure is as follows:
```
/chest_xray
    /train
        /NORMAL
        /PNEUMONIA
    /test
        /NORMAL
        /PNEUMONIA
    /val
        /NORMAL
        /PNEUMONIA
```

You can download the dataset from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) or other public sources.

## Project Structure

The repository is structured as follows:
```
├── data/                       # Contains training, validation, and test data
├── models/                     # Contains the trained models
├── notebooks/                  # Jupyter notebooks for data exploration and model development
├── scripts/                    # Python scripts for model training and evaluation
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
```

### Key Files
- **train.py**: Script to train the CNN model.
- **predict.py**: Script to predict whether an X-ray image has pneumonia or not using a trained model.
- **model.py**: Defines the CNN architecture.
- **utils.py**: Helper functions for preprocessing and model evaluation.
- **notebooks/**: Contains Jupyter notebooks for EDA (Exploratory Data Analysis) and training the model.

## Model Architecture

We implemented a Convolutional Neural Network (CNN) using Keras and TensorFlow to classify the X-ray images. The model consists of the following layers:

- Convolutional layers with ReLU activation.
- MaxPooling layers for down-sampling.
- Dropout layers to prevent overfitting.
- Fully connected layers with a Softmax output.

The architecture was fine-tuned through experimentation to achieve the best performance on the dataset.

## Results

Our final model achieved the following metrics:
- **Training Accuracy**: 98%
- **Validation Accuracy**: 90%
- **Test Accuracy**: 88%

We have included the confusion matrix, precision, recall, and F1-score for further analysis in the Jupyter notebook files.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lasithadilshan/Chest_XRay_Pneumonia_Project.git
   cd Chest_XRay_Pneumonia_Project
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and extract the dataset** into the `data/` directory, as per the directory structure shown above.

## Usage

### Training the Model

To train the model from scratch, run:
```bash
python scripts/train.py
```
This script will preprocess the dataset and train the CNN model. The trained model will be saved in the `models/` directory.

### Making Predictions

To make predictions on new X-ray images, run:
```bash
python scripts/predict.py --image_path <path_to_image>
```
The script will load the trained model and output the classification result (Normal/Pneumonia).

### Jupyter Notebooks

Explore the Jupyter notebooks in the `notebooks/` directory to understand the data analysis, model architecture, training process, and evaluation.

## Contributing

Contributions are welcome! If you have suggestions or want to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Push to your fork and submit a pull request.

Please ensure your code adheres to the existing style and includes necessary tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Special thanks to the contributors of the [Kaggle Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) for making this project possible.
- This project was inspired by various research papers and open-source deep learning projects in the healthcare domain.
- Gratitude to all contributors and users who support open-source initiatives in medical AI.

---
