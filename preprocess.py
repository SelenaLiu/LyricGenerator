from utils import Utils
import pickle, os, subprocess

class Preprocess():

    def __init__(self):
        self.ut = Utils()
        self.saved = "saved.p"

    def get_max_length(self):
        #needs code
        #gets longest song from entire array
        pass

    def equalize_length(self):
        #needs code
        #pads all songs at end to be same size as max length
        pass

    def mp3_to_wav(self,audio_path):
        if(os.path.exists(self.saved)):
            converted = pickle.load(open(self.saved,"rb"))
        else:
            converted = []
        for file in os.listdir(audio_path):
            if file.endswith(".mp3") & (file not in converted):
                print("Converting "+file+" from mp3 to wav")
                subprocess.call(["ffmpeg","-i",(os.path.join(audio_path,file)),(os.path.join(audio_path,self.ut.change_ext(file,".wav")))])
                converted.append(file)
        pickle.dump(converted,open(self.saved,"wb"))
