import os

root_folder=os.getcwd()

audio_data_folder=os.path.join(root_folder,"..\\")

metadata_folder=os.path.join(root_folder,"..\\UrbanSound8K.csv")

MFCC_folder=os.path.join(root_folder,'MFCC')
if not os.path.exists(MFCC_folder):
    os.mkdir(MFCC_folder)

Mel_folder=os.path.join(root_folder,'Mel')
if not os.path.exists(Mel_folder):
    os.mkdir(Mel_folder)

for i in range(1,11):
    fold_folder_MFCC=os.path.join(MFCC_folder,'fold_'+str(i))
    if not os.path.exists(fold_folder_MFCC):
        os.mkdir(fold_folder_MFCC)
    fold_folder_Mel = os.path.join(Mel_folder, 'fold_' + str(i))
    if not os.path.exists(fold_folder_Mel):
        os.mkdir(fold_folder_Mel)


batch_size=32
init_lr = 1e-3
early_stopping_patience = 15
reduce_lr_patience = 8

input_size=(1,13,344,1) # MFCC-mono
#input_size=(1,64,344,1) # Mel - mono
