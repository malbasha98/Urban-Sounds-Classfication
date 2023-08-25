Classification of audio files from the UrbanSounds8k dataset using the ResNet18 architecture.

## Description:
The system consists of the following scripts:
- `settings.py`
    - This script creates the "MFCC" and "Mel" folders (each containing folders named "fold_i" where "i" ranges from 1 to 10). These folders are used to store PNG training graphs, models, and training history. (history). 
- `AudioPrep.py`
    - This script performs preprocessing of audio files and extracts descriptors (you can choose between MFCC and MelSpectrogram).
- `SoundDS.py`
    - This script contains auxiliary functions for loading the audio dataset and extracting descriptors from its audio files.
- `data.py`
    - This script loads the UrbanSounds8k audio dataset and extracts descriptors from it.   
- `model.py`
    - This script creates the ResNet18 model.
- `train.py`
    - This script is used for training and evaluating the model on the UrbanSounds8k dataset.

## Usage
First, download the UrbanSounds8k dataset from Kaggle, which contains the following folders:
    - /fold1
    - /fold2
    - /fold3
    - /fold4
    - /fold5
    - /fold6
    - /fold7
    - /fold8
    - /fold9
    - /fold10
    - /UrbanSound8k.csv
The directory structure should be set up as follows:
- ./
    - /fold1
    - /fold2
    - /fold3
    - /fold4
    - /fold5
    - /fold6
    - /fold7
    - /fold8
    - /fold9
    - /fold10
    - /PycharmProject
    - /UrbanSound8k.csv
      
To perform audio file classification using the ResNet18 architecture on the UrbanSounds8k dataset, follow these steps:

- Run the settings.py script to create folders for storing model checkpoints and training plots. This script will create the MFCC and Mel folders, each containing subfolders fold_i (where i ranges from 1 to 10).

- Run the data.py script to create a dataframe of extracted features from the UrbanSounds8k dataset. This script will load the audio dataset and extract the desired descriptors.

- Run the model.py script to create a ResNet18 model. This script defines the architecture and initializes the model.

- Run the train.py script to train the model and evaluate its performance. This script will use the preprocessed data and the ResNet18 model to train the classifier on the UrbanSounds8k dataset.

## Results

- An accuracy of 72%-73% was achieved using Mel Spectrograms, while an accuracy of 68%-70% was achieved using MFCCs.

