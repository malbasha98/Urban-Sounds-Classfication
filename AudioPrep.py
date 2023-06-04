import librosa
import random
import numpy as np

class AudioPrep:
    @staticmethod
    def open_audio_file(audio_file):
        sig, sr=librosa.load(audio_file, mono=False)
        return (sig, sr)
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud
        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud
        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig=librosa.to_mono(sig)
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = librosa.util.stack([sig, sig])
        return ((resig, sr))
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud
        if (sr == newsr):
            # Nothing to do
            return aud
        num_channels = sig.shape[0]
        # Resample first channel
        if (num_channels > 1 and len(sig.shape)>1):
            # Resample the second channel and merge both channels
            reone = librosa.resample(y=sig[0,:],orig_sr= sr,target_sr= newsr)
            retwo = librosa.resample(y=sig[1,:],orig_sr= sr,target_sr= newsr,)
            resig = librosa.util.stack([reone, retwo])
        else:
            resig=librosa.resample(sig, orig_sr=sr, target_sr=newsr)
        return ((resig, newsr))
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        sig_len=0
        if (len(sig.shape)>1):
            sig_len = sig.shape[1]
        else:
            sig_len=len(sig)
        max_len = sr//1000 * max_ms
        if (sig_len > max_len or sig_len < max_len):
            # Truncate the signal to the given length
            sig=librosa.util.fix_length(sig, size=max_len)
        return (sig, sr)
    @staticmethod
    def time_shift(aud, shift_limit):
        sig,sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
    @staticmethod
    def mel_spectrogram(aud, n_mels=64, n_fft=1024):
        sig,sr = aud
        top_db = 80
        spec=librosa.feature.melspectrogram(y=sig, sr=sr,n_mels=n_mels, n_fft=n_fft)
        spec=librosa.power_to_db(spec, ref=np.max)
        return spec
    @staticmethod
    def mfccs(aud, n_mfcc=13):
        sig,sr = aud
        spec = librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n_mfcc)
        return spec