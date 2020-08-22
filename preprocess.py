from utils import Utils
import numpy as np
import pickle, os, subprocess

class Preprocess():

    def __init__(self,sr):
        self.ut = Utils()
        self.converted_audio = "conv_aud.p"
        self.saved_audio_np = "saved_np.p"
        self.sample_rate = sr
        self.max_length = 0

    def get_max_length(self):
        return self.max_length

    def process_data(self, list_songs: List):
        # gets longest song from entire array and pads all songs at end to be
        # same size as max length
        list_songs.sort(key=len)
        self.max_length = list_songs[-1].shape[0]
        for element in list_songs:
            element_len = element.shape[0]
            N = self.max_length - element_len
            np.pad(element, (0, N), 'constant')

    def mp3_to_wav(self,audio_path):
        if(os.path.exists(self.converted_audio)):
            converted = pickle.load(open(self.converted_audio,"rb"))
        else:
            converted = []
        for file in os.listdir(audio_path):
            if file.endswith(".mp3") & (file not in converted):
                print("Converting "+file+" from mp3 to wav")
                subprocess.call(["ffmpeg","-i",(os.path.join(audio_path,file)),(os.path.join(audio_path,self.ut.change_ext(file,".wav")))])
                converted.append(file)
        pickle.dump(converted,open(self.converted_audio,"wb"))
        return converted

    def compile_audio(self,audio_path):
        if(os.path.exists(self.saved_audio_np)):
            save = pickle.load(open(self.saved_audio_np,"rb"))
            return save
        else:
            save = []
        for file in os.listdir(audio_path):
            if file.endswith(".wav"):
                signal,sr = self.ut.load_audio(os.path.join(audio_path,file),self.sample_rate)
                save.append(signal)
        pickle.dump(save,open(self.saved_audio_np,"wb"))
        return save

    # def equalize_length():
    #     #needs code
    #     #pads all songs at end to be same size as max length
    #     pass
