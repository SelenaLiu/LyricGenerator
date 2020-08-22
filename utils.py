import librosa

class Utils():

    def __init__():

    def load_audio(self,file_path,target_sr):
        signal, sample_rate = librosa.load(file_path,sr=target_sr,mono=True)
        return signal, sample_rate
