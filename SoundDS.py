from settings import *
from AudioPrep import AudioPrep

class SoundDS:
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_file = self.data_path + str(self.df["fold"][idx]) + '/' + self.df["slice_file_name"][idx]
        class_id = self.df["classID"][idx]
        aud = AudioPrep.open_audio_file(audio_file)
        rechan = AudioPrep.rechannel(aud, self.channel)
        reaud = AudioPrep.resample(rechan, self.sr)
        dur_aud = AudioPrep.pad_trunc(reaud, self.duration)
        # sgram = AudioPrep.spectro_gram(dur_aud,n_fft=1024)
        mfcc_features = AudioPrep.mfccs(dur_aud)
        fold = self.df['fold'][idx]
        return mfcc_features, class_id, fold