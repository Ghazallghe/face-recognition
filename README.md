# face-recognition

This project implements face recognition using a small dataset of Iranian actors and actresses through three different methods:
1. A custom-built CNN (Convolutional Neural Network) model from scratch.
2. Transfer learning using the ResNet architecture pre-trained on ImageNet.
3. Transfer learning using a custom model trained on a different dataset of foreign actors.

**Note**: The final model weights used for training have been included in the repository.

## Project Overview

The project includes training and validating models on a dataset of Iranian actors. The models are built using TensorFlow and Keras. The three approaches used for this task are:

### 1. Custom CNN Model

- **Network Structure**: We designed and implemented a custom convolutional neural network from scratch, tailored for the face recognition task.
- **Preprocessing and Data Augmentation**: We applied several augmentation techniques such as random flipping, rotation, zoom, contrast adjustment, and brightness modification to enhance the model's generalization.

### 2. Transfer Learning with ResNet

- **Architecture**: We used the ResNet101 architecture pre-trained on ImageNet.
- **Data Preprocessing**: Standard preprocessing and augmentation techniques, including random flips and rotations, were applied.
- **Training**: The base model was fine-tuned for face recognition using the Iranian actors dataset.

### 3. Transfer Learning Using Custom Model Weights

- **Pre-trained Model**: A custom face recognition model was trained on a different dataset of foreign actors.
- **Transfer Learning**: The weights from this model were used to fine-tune the network for the Iranian actors dataset.
- **Architecture**: The architecture of the custom model was used, with the classification layers retrained for the new dataset.

## Dataset

The dataset consists of images of Iranian actors, which were split into 80% for training and 20% for validation. The input image size was set to `(160, 160)` and the batch size to 32.

## Notebooks
The project contains three Jupyter notebooks:

1. `face_recognition.ipynb`: Implements all three methods, including the custom CNN, ResNet-based transfer learning, and the custom transfer learning approach.

2. `transfer_learning.ipynb`: Focuses on the custom transfer learning method, where the model trained on a foreign actors dataset is fine-tuned for the Iranian actors dataset.

3. `utils.ipynb`: Contains utility functions for plotting images, as well as visualizing the training/validation loss and accuracy.

## Key Components

### 1. Data Augmentation

We used several augmentation techniques to improve model robustness and reduce overfitting. These included:

- Random flip
- Random rotation
- Random zoom
- Random contrast adjustment
- Random brightness
- Random translation

### 2. Model Architectures

- **Custom Model**: 
  - Consists of convolutional layers followed by max-pooling, flattening, and dense layers for classification.
  - Uses L2 regularization and dropout to avoid overfitting.

- **Transfer Learning with ResNet**:
  - Base model: ResNet101 with ImageNet weights, excluding the top classification layer.
  - The pre-trained layers are frozen initially, and only the classification layers are trained.

- **Transfer Learning from Custom Model**:
  - The base model (pre-trained on a foreign actors dataset) is frozen, and the classification layers are retrained on the Iranian actors dataset.
