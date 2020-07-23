import librosa
import numpy as np
import IPython.display as ipd
import matplotlib.pyplot as plt

class WavAugmentation(object):
    def __init__(self, path):
        super(WavAugmentation, self).__init__()
        self.wav, self.sr = self.load_wav(path)

    def load_wav(self, path):
        data, sr = librosa.load(path)
        return data, sr

    def add_noise(self):
        noise = np.random.randn(len(self.wav))
        data_noise = self.wav + 0.002*noise
        return data_noise

    def shift(self):
        return np.roll(self.wav, self.sr)

    def stretch(self, rate=1):
        data = librosa.effects.time_stretch(self.wav, rate)
        return data

    def register_wav(self, path, data):
        librosa.output.write_wav(path, data, self.sr)

    def plot_time_series(self, data):
        fig = plt.figure(figsize=(14, 8))
        plt.title('Raw wave ')
        plt.ylabel('Amplitude')
        plt.plot(np.linspace(0, 1, len(data)), data)
        plt.show()