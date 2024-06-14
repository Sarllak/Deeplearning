
# Emotion Classification from Audio Files

## Overview
This project involves processing and classifying audio files to identify various emotions through the analysis of WAV files. The goal of this project is to create a deep learning model for classifying emotions in audio files using features extracted from the audio such as pitch shifting and time stretching, as well as using MFCC and energy features.

## Models

In this project, two different approaches were considered to compare how well it classifies emotions based on audio files given. The first approach is image classification using 2D CNN on spectrograms. Using the Librosa library, convert .wav files to respective mel-spectrogram and use them to train 2D CNN model. The second approach is sequence modeling on MFCCs using CNN-LSTM model structure. Convert .wav files into a sequence of MFCC and use it to train CNN-LSTM model.

### 2D CNN
The model has three 2D convolutional layers with 32 filters followed by two fully connected layers with 128 and 7 nodes and an output layer with 7 nodes and softmax activation. Dropout layers were used throughout the model to prevent overfitting.

### CNN-LSTM
The model has two 2D convolutional layers with 32 filters followed by one LSTM layer with 100 nodes, then followed by one fully connected layer of 128 nodes and an output layer of 7 nodes and softmax activation. Dropout layers were used throughout the model to prevent overfitting.

## Results
Both models had similar performance, with CNN-LSTM model slightly outperforming CNN. More rigorous hyperparameter tuning with different model structure and more training epochs may increase the overall accuracy of these models.

| Model      | Loss  | Accuracy |
|------------|-------|----------|
| CNN        | 0.2136 | 94.33%   |
| CNN-LSTM   | 0.2633 | 93.64%   |

## Instructions
1. Load the `ipynb` file.
2. Follow the code cells to understand the preprocessing, augmentation, and model training steps.
3. Evaluate the results and compare the performance of the two models.
