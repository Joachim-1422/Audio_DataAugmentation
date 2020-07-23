import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import os

class SoundFilter(object):
	def __init__(self,l_filters=[]):
		super().__init__()
		self.equiv = {
			'high':self.highPassFilter,
			'low':self.lowPassFilter
		}
		self.filters = l_filters
		self.buildPipeline()

	def buildPipeline(self,):
		if len(self.filters) <= 0:
			self.pipeline = []
		self.pipeline = [
			self.equiv[i] for i in self.filters if i in self.equiv.keys() ]

	def apply(self,data):
		for app in self.pipeline:
			data = app(data)
		return data

	def lowPassFilter(self,data):
		fc = 0.1
		b = 0.08
		N = int(np.ceil((4 / b)))

		if not N % 2:
			N += 1
		n = np.arange(N)

		sf = np.sinc(2 * fc * (n - (N - 1) / 2.))
		window = np.blackman(N)
		sf = sf * window
		sf = sf / np.sum(sf)

		return np.convolve(data, sf)

	def highPassFilter(self,data):
		fc = 0.1
		b = 0.08
		N = int(np.ceil((4 / b)))

		if not N % 2:
			N += 1
		n = np.arange(N)

		sf = np.sinc(2 * fc * (n - (N - 1) / 2.))
		window = np.blackman(N)
		sf = sf * window
		sf = sf / np.sum(sf)

		sf = -sf
		sf[int((N - 1) / 2)] += 1

		return np.convolve(data, sf)

class CsvWriter(object):
	def __init__(self,outPath):
		super().__init__()
		self.out = open(outPath,'w')

	def __del__(self,):
		self.out.close()

	def LineWriter(self,data,index=None,sep=';'):
		self.out.write(index+';'+sep.join([ str(i) for i in data ])+'\n')\
			if index is not None else\
				self.out.write(sep.join([ str(i) for i in data ])+'\n')

	def dataWriter(self,data,index=None,sep=';'):
		for line in data:
			self.LineWriter(line,index,sep)

class ExtractSoundFeature(object):
	def __init__(self,filters):
		super().__init__()
		self.filters = filters

	def extract(self,path):
		y,t_sr = librosa.load(path)
		y = self.filters.apply(y)
		chroma_stft = librosa.feature.chroma_stft(y=y, sr=t_sr)
		rms = librosa.feature.rms(y=y)
		spec_cent = librosa.feature.spectral_centroid(y=y, sr=t_sr)
		spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=t_sr)
		rolloff = librosa.feature.spectral_rolloff(y=y, sr=t_sr)
		zcr = librosa.feature.zero_crossing_rate(y)
		mfcc_extract = librosa.feature.mfcc( y=y,sr=t_sr,n_mfcc=39 )
		return	[np.mean(i) for i in [ chroma_stft,rms,spec_cent,spec_bw,rolloff,zcr ]]\
			+ [ np.mean(i) for i in mfcc_extract ]

from threading import Thread

class CreateVoiceDataset(Thread):
	def __init__(self,path,outPath,isHighPass,tid):
		super().__init__()
		self.path = path
		self.tid = tid
		t_filters = SoundFilter(
			l_filters=['high'])\
		if isHighPass is True else\
			SoundFilter()

		self.writer = CsvWriter(outPath)
		self.extractor = ExtractSoundFeature(
			filters=t_filters)

		self.columns = [
			'filename',
			'chroma_stft',
			'rms',
			'spectral_centroid',
			'spectral_bandwidth',
			'rolloff',
			'zero_crossing_rate'] + [
				'mfcc'+str(i) for i in range(39)] + [
					'target','num_actor']

		self.loadFilenames()

	def loadFilenames(self,):
		if type(self.path) == list:
			self.datapath = []
			for t_path in self.path:
				for t_dir,_,filenames in os.walk(t_path):
					for  i in filenames:
						if '.wav' in i:
							self.datapath.append( os.path.join(t_dir,i) )
		else:
			self.datapath = []
			for t_dir,_,filenames in os.walk(self.path):
				for  i in filenames:
					if '.wav' in i:
						self.datapath.append( os.path.join(t_dir,i) )

	def run(self):
		print(self.tid,'Loading dataset...')
		self.writer.LineWriter(self.columns)

		for it,path in enumerate(self.datapath):
			print(self.tid,'State:', '{}/{}'.format(it, len(self.datapath)))
			sound_features = self.extractor.extract(path)
			t_extract_target = path.split('-')
			sound_features+=[int(t_extract_target[2])-1,int(t_extract_target[6].split('.')[0]) ]

			self.writer.LineWriter(sound_features, index=path)
		print(self.tid,'dataset loaded.')

if __name__ == '__main__':
	t_pool = []

	#With High pass filter
	t_pool.append(
		CreateVoiceDataset(
			'./Data', './Datasets/highpass_dataset.csv', isHighPass=True, tid=0))
	#Without High pass filter
	t_pool.append(
		CreateVoiceDataset(
			'./Data', './Datasets/base_dataset.csv', isHighPass=False, tid=1))
	#With High pass filter and augmentation
	t_pool.append(
		CreateVoiceDataset(
			['./Data','./Augmentation'], './Datasets/aug_highpass_dataset.csv', isHighPass=True, tid=2))
	#Without High pass filter and augmentation
	t_pool.append(
		CreateVoiceDataset(
			['./Data','./Augmentation'], './Datasets/aug_base_dataset.csv', isHighPass=False, tid=3))

	#Start threads
	[ i.start() for i in t_pool ]
	#Wait threads each other
	[ i.join() for i in t_pool ]
