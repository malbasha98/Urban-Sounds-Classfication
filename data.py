import pandas as pd
from datetime import datetime
from settings import *
import numpy as np
from SoundDS import SoundDS
from tensorflow import keras
from keras.utils import to_categorical

def load_data():
    metadata=pd.read_csv(metadata_folder)
    audioDS=SoundDS(metadata, audio_data_folder+"/fold")

    features=[]
    folds=[]
    labels=[]
    loaded_item_num=0
    for i in range(len(metadata)):
        feature, label, fold=audioDS.__getitem__(i)
        features.append(feature)
        labels.append(label)
        folds.append(fold)
        print("\r", loaded_item_num +i+1, "/8732", end="")
    print("\r\n")
    folds_np = np.array(folds)
    features_np = np.array(features)
    labels_np = np.array(labels)
    return features_np, labels_np, folds_np

print("\n-----Data preparation-----\n")
start = datetime.now()
features, labels, folds=load_data()
labels=to_categorical(labels)
duration = datetime.now() - start
print("\n-----Data ready for training (preparation time ", duration, ")-----\n")