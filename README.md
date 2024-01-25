# Speech-Emotion-Recognition
A model that can predict the embedded emotions in an audio signal, using convolutional neural network.
# Introduction
## Problem statement
It is required to build a model that can predict the embedded emotions in an audio signal. This is a speech emotion recognition model to predict human emotions using the convolutional neural network (CNN) by learning segmented audio of specific emotions. 

# Dataset Description
## Overview
Dataset used: CREMA. It is a dataset that consists of 7442 audio files. Each file represents a specific emotion. 
Six emotions are available:
1.	Sadness
2.	Fear
3.	Happiness
4.	Anger
5.	Disgust
6.	Neutral

# Audio Preprocessing
According to the paper entitled ‘Preprocessing Signal for Speech Emotion Recognition’:
Before the extraction of the features of the signal, this signal is manipulated by using preprocessing. 
Following the paper ideas, we implemented some of the preprocessing operations that are proposed on the audio signals. 
## Noise Reduction
Any audio signal has noise due to the surrounding environment. Removing this noise will eventually improve the accuracy of the model
Implemented Function: spectral_subtraction
Purpose/Description: The purpose of the spectral_subtraction function is to remove background noise from the given audio using the spectral subtraction technique.
## Silence Removal
The speech signal usually includes many parts of silence. The silence signal is not important because it is does not contain information.
Implemented Function: remove_silence
Purpose/Description: The purpose of the remove_silence function is to remove the silent sections from the given audio data.
## Pre-emphasis
The pre-emphasis of the speech signal is the most important steps of preprocessing at high frequency. It's used to get comparable amplitude. To fulfill the assignment, the speech signal is passed through a high-pass filter (FIR).
Implemented Function: pre-emphasis
Purpose/Description: The purpose of the preemphasis function is to apply preemphasis filtering to the given audio signal. Preemphasis is a technique used to emphasize higher frequencies and enhance certain aspects of the signal, such as improving speech intelligibility or aiding in audio processing tasks.
## Normalization
Implemented Function: normalize_audio
Purpose/Description: The purpose of the normalize_audio function is to normalize the given audio signal by subtracting the mean and dividing by the standard deviation. Normalization is a common preprocessing step used to scale the values of the signal to a standardized range.
## Truncate and Pad
Audio signals in CREMA dataset are not of the same length. CNN needs to take input of same length. Many approaches can be taken to solve this problem. Either to truncate audio to the minimum file duration, or to decide to pad all audio signals with zeros to make them equal to the longest file. We decided to take another approach, which is to truncate audio file at a specific duration, and files with length less than that duration, padding 
Implemented Function: truncate_pad
Purpose/Description: The purpose of the truncate_pad function is to modify the given audio data to match the desired duration by either truncating or padding the data.
# Feature Spaces
We will create two feature spaces from the audio. First, we will begin with time domain and frequency domain and build a 1D convolution model. Second, audio files will be converted to spectrograms, and a 2D convolution model will be used.
## Features for 1D convolution
### Energy
Energy is a commonly used feature in audio signal processing and can be useful for signal emotion recognition. The energy of an audio signal represents the overall strength or power of the signal and provides information about the magnitude of the sound.

Implemented Function: energy
Purpose/Description: The purpose of the energy function is to compute the energy of an audio signal using a frame-based analysis approach. The function calculates the energy of each frame by summing the squared magnitude of the audio data within each frame.

### Entropy of Energy
Function Name: entropy_of_energy
Purpose/Description: The purpose of the entropy_of_energy function is to compute the entropy of the energy distribution of an audio signal. The function first calculates the energy values of each frame using the energy function. Then, it normalizes the energy values and computes the entropy based on the energy distribution.

### Entropy of Energy
Function Name: entropy_of_energy
Purpose/Description: The purpose of the entropy_of_energy function is to compute the entropy of the energy distribution of an audio signal. The function first calculates the energy values of each frame using the energy function. Then, it normalizes the energy values and computes the entropy based on the energy distribution.

### Zero Crossing Rate
Function Name: zero_crossing_rate
Purpose/Description: The purpose of the zero_crossing_rate function is to compute the zero-crossing rate (ZCR) of an audio signal. The ZCR represents the rate at which the audio signal changes its sign (from positive to negative or vice versa) over time. It is a measure of the frequency content and temporal characteristics of the signal.

### Root Mean Square
Purpose/Description: The purpose of the rmse function is to compute the Root Mean Square Energy (RMSE) of an audio signal. The RMSE represents the average magnitude of the audio signal over time and provides information about the overall energy or loudness.

### Zero Crossing Rate
Function Name: mfcc
Purpose/Description: This function computes the Mel-frequency cepstral coefficients (MFCCs) of an audio signal.

## Spectrogram (2D convolution)
A spectrogram is a visual representation of the frequency content of a signal over time. It provides a way to analyze and visualize how the spectral components of a signal change over different time intervals.
In a spectrogram, the x-axis represents time, the y-axis represents frequency, and the intensity of the color or shading represents the magnitude or power of the corresponding frequency component. Darker or more intense colors typically indicate higher energy or power at specific frequencies.

# Data Augmentation
Data augmentation is a technique used in machine learning, specifically in the context of training convolutional neural networks (CNNs), to artificially increase the size and diversity of the training dataset by applying various transformations or modifications to the existing data.
The primary goal of data augmentation is to introduce variability and enhance the generalization capability of the CNN model. 
We implemented three functions for data augmentation which are explained in the following sections.
## Noise
The noise function adds random noise to the audio data. It calculates the amplitude of the noise based on a fraction of the maximum amplitude of the data. It then adds this noise to the audio by generating random values from a normal distribution. Adding noise helps the model generalize better to real-world scenarios with background noise, enhances robustness, and acts as a form of regularization.
## Shift
The shift function performs a random shifting of the audio data by a certain number of samples. The shift range is determined by a random value between -5 and 5, multiplied by 2000. Shifting the audio in time can simulate temporal variations and help the model learn invariant features. It also provides robustness to slight time misalignments between the training data and test data.    
## Pitch
The pitch function applies pitch shifting to the audio data. Pitch shifting changes the pitch or frequency of the audio without affecting its duration. It is achieved by modifying the time-frequency representation of the audio. The pitch factor is a parameter that determines the amount of pitch shift to be applied. Pitch augmentation can help the model handle variations in pitch, such as different speakers or musical instruments, and improve its generalization capabilities.

# Model 1
So before explaining our model in detail let’s summarize. As mentioned, preprocessing operations must be performed on audio signals before training. Also, we need to split our dataset intro training, validating, and testing set.
That’s why, first the data is split in to three separated sets while maintaining the labels in vectors. The labels here are the emotion associated with each file. 
After splitting the data, the training set must be augmented for better training. The three augmentation operations mentioned are applied to training data.
After data augmentation, features are extracted from the entire datasets. That’s why in our code, you find functions called: ‘prepare_train_data’ and ‘prepare_test’. These functions have the objective of extracting the features from given files and return the sets that will be used for training.
Finally, after data augmentation and feature extraction, the CNN begins its job.

## Accuracy
We reached training accuracy of 96.14 %.
Validation accuracy of 50.57 %.
Finally, testing accuracy is 52.80 %.

# Model 2 (Spectrograms)
The difference is that the input to the CNN here is a spectrogram. That’s why we use 2D convolutions. No features are extracted from audio here, only spectrograms are used.

## Accuracy
We reached training accuracy of 99.95 %.
Validation accuracy of 54.79 %.
Finally, testing accuracy is 55.40 %.
 
# Papers Used
1.  Preprocessing Signal for Speech Emotion Recognition, 2017
2. Speech emotion recognition using 2D-convolutional neural network, 2022
