import librosa, os

class Utils():

    def __init__(self):
        pass

    def load_audio(self,file_path,target_sr):
        signal, sample_rate = librosa.load(file_path,sr=target_sr,mono=True)
        samples = librosa.resample(signal,sample_rate,target_sr)
        return samples, sample_rate

    def change_ext(self,filename,out):
        return os.path.splitext(filename)[0]+out
