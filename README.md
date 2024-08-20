### README

# Face Image Classification with CNN

This project is focused on classifying facial images into different categories using a Convolutional Neural Network (CNN). The dataset comprises various images of faces, each associated with specific attributes like pose, expression, and eye status. The project includes steps for data preprocessing, augmentation, model training, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Model Evaluation](#model-evaluation)
- [Results and Insights](#results-and-insights)
- [Next Steps](#next-steps)
- [License](#license)

## Overview

This project implements a Convolutional Neural Network (CNN) for the classification of facial images. The images are preprocessed, resized, and normalized before being augmented to create a robust training dataset. The CNN model is trained and evaluated to classify the images based on their attributes.

## Installation

1. **Clone the repository:**
   ```bash
   git clone repository_url
   ```

2. **Install the required libraries:**
   ```bash
   pip install tensorflow pandas requests pillow numpy scikit-learn
   ```

3. **Run the script:**
   - Ensure that the script is executed in an environment with internet access to fetch the dataset.

## Data Preparation

1. **Data Collection:**
   - The dataset is composed of facial images fetched from a remote source. Each image is resized to a consistent size of 64x60 pixels and normalized to a pixel value range between 0 and 1.

2. **Label Encoding:**
   - The labels are extracted from the filenames and encoded using one-hot encoding for multi-class classification.

3. **Data Splitting:**
   - The dataset is split into training, validation, and test sets to train and evaluate the model's performance.

## Data Augmentation

- Data augmentation is applied to the training dataset using the `ImageDataGenerator` from TensorFlow. This includes transformations like random rotations, shifts, shear, zoom, and horizontal flips, which help increase the variability of the training data.

## Model Architecture

- The CNN model is built using the following architecture:
  - **Conv2D Layer 1:** 32 filters, 3x3 kernel, ReLU activation
  - **MaxPooling Layer 1:** 2x2 pool size
  - **Conv2D Layer 2:** 64 filters, 3x3 kernel, ReLU activation
  - **MaxPooling Layer 2:** 2x2 pool size
  - **Conv2D Layer 3:** 64 filters, 3x3 kernel, ReLU activation
  - **Flatten Layer**
  - **Dense Layer 1:** 64 units, ReLU activation
  - **Dense Layer 2:** 20 units (for 20 classes), Sigmoid activation

## Training the Model

- The model is compiled with the Adam optimizer and categorical crossentropy loss. It is trained over 10 epochs with a batch size of 32, using accuracy as the primary metric.

## Model Evaluation

- After training, the model is evaluated on the test set to assess its performance. The evaluation metrics include accuracy and loss.

## Results and Insights

- **Training Results:**
  - The model shows steady improvement over 10 epochs, with accuracy increasing from 10.64% to 97.49%, and validation accuracy reaching 92.31%.

- **Validation Loss and Accuracy:**
  - The validation loss decreased consistently across epochs, indicating that the model was learning effectively and generalizing well to unseen data.

- **Test Set Evaluation:**
  - The final model achieved an impressive accuracy of 98.10% on the test set, with a loss of 0.0585, suggesting that the model is highly accurate in classifying the images.

- **Model Performance:**
  - The significant improvement in accuracy and reduction in loss across epochs indicate that the model architecture and data augmentation techniques were effective in enhancing model performance.

## Next Steps

- **Increase Dataset Size:**
  - Experiment with a larger and more diverse dataset to further improve model robustness.

- **Hyperparameter Tuning:**
  - Conduct hyperparameter tuning to optimize the model further, possibly increasing the number of filters, changing the activation functions, or adjusting the learning rate.

- **Advanced Model Architectures:**
  - Explore more complex architectures such as ResNet or InceptionNet to see if they provide better accuracy on this dataset.

- **Transfer Learning:**
  - Implement transfer learning by leveraging pre-trained models to potentially boost performance with fewer epochs.

## License

This project is licensed under the MIT License.

---

### Insights on Epoch Results and Model Evaluation

- **Epoch Results:**
  - The model demonstrates a strong learning curve with rapid improvement in the early epochs. Starting with an accuracy of 10.64% (close to random guessing for 20 classes), the model quickly improves to 97.49% by the 10th epoch. The reduction in loss from 2.85 to 0.08 signifies that the model is not only learning but also refining its predictions with each epoch.

- **Model Evaluation:**
  - The final test accuracy of 98.10% reflects the model's strong generalization ability, showing that it can accurately classify unseen data. The low test loss of 0.0585 further indicates that the model is not overfitting and has maintained good performance outside the training data.

- **Conclusion:**
  - Overall, the results suggest that the CNN architecture used in this project is highly effective for facial image classification within the given dataset. Future enhancements could include experimenting with more sophisticated models or using transfer learning to handle even more complex and varied datasets.
