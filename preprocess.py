import numpy as np
import DALI as dali_code

import pickle, os, subprocess, librosa

class Preprocess():

    def __init__(self,sr,batch_size):
        self.converted_audio = "conv_aud.p"
        self.saved_audio_np = "saved_np.p"

        self.dataset_size = 5358
        self.sample_rate = sr
        self.max_length = 14553000 # Changed from 0

        self.data_dir = "/mnt/d/Repositories/LyricGenerator/data"
        self.audio_dir = os.path.join(self.data_dir,"Audio")
        self.lyric_dir = os.path.join(self.data_dir,"Lyrics")
        self.json_dir = os.path.join(self.data_dir,"JSON")
        self.dali_dir = os.path.join(self.data_dir,"DALI_v1.0")

    def load_audio(self,file_path,target_sr):
        signal, sample_rate = librosa.load(file_path,sr=target_sr,mono=True)
        samples = librosa.resample(signal,sample_rate,target_sr)
        return samples, sample_rate

    def change_ext(self,filename,out):
        return os.path.splitext(filename)[0]+out

    def find_max_length(self):
        max_duration = 0
        for file in os.listdir(self.audio_dir):
            if file.endswith(".wav"):
                duration = librosa.get_duration(filename=os.path.join(self.audio_dir,file),sr=self.sample_rate)
                if(duration>max_duration):
                    max_duration=duration
        return max_duration

    def process_data(self, list_songs):
        # gets longest song from entire array and pads all songs at end to be
        # same size as max length
        list_songs.sort(key=len)
        # self.max_length = list_songs[-1].shape[0]
        for element in list_songs:
            element_len = element.shape[0]
            N = self.max_length - element_len
            np.pad(element, (0, N), 'constant')

    def mp3_to_wav(self):
        if(os.path.exists(self.converted_audio)):
            converted = pickle.load(open(self.converted_audio,"rb"))
        else:
            converted = []
        for file in os.listdir(self.audio_dir):
            if file.endswith(".mp3") & (file not in converted):
                wav_filename = self.change_ext(file,".wav")
                if(os.path.exists(os.path.join(self.audio_dir,wav_filename))):
                    print("Skipping "+wav_filename+", already converted.")
                    os.remove(file)
                else:
                    print("Converting "+file+" from mp3 to wav")
                    subprocess.call(["ffmpeg","-i",(os.path.join(self.audio_dir,file)),(os.path.join(self.audio_dir,wav_filename))])
                converted.append(file)
        pickle.dump(converted,open(self.converted_audio,"wb"))
        return converted

    def compile_audio(self):
        if(os.path.exists(self.saved_audio_np)):
            save = pickle.load(open(self.saved_audio_np,"rb"))
            return save
        else:
            save = []
        for file in os.listdir(self.audio_dir):
            if file.endswith(".wav"):
                print("Loading "+file)
                signal,sr = self.load_audio(os.path.join(self.audio_dir,file),self.sample_rate)
                save.append(signal)
        pickle.dump(save,open(self.saved_audio_np,"wb"))
        return save

    def find_files(self, filename, search_path):
        result = []
        # Walking top-down from the root
        for root, dir, files in os.walk(search_path):
            if filename in files:
                result.append(os.path.join(root, filename))
        return result

    def dali_json_to_np(self):
        dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
        dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")

        all_songs = []
        batch_counter = 0

        for i in range(1, self.dataset_size+1):
            entry = dali_data[dali_info[i].item(0)]
            if (len(self.find_files(dali_info[i].item(0) + ".wav",self.audio_dir))!= 0):
                words = get_single_lyric(entry)
                words_np = np.array(words)
            all_songs.append(words_np)
        return np.array(all_songs)

    def dali_json_to_txt(self):
        dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
        dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")

        for i in range(1, self.dataset_size+1):
            entry = dali_data[dali_info[i].item(0)]
            if (len(self.find_files(dali_info[i].item(0) + ".wav",self.audio_dir))!= 0):
                # print(entry.annotations["annot"].keys())
                words = get_single_lyric(entry)
                with open(os.path.join(self.lyric_dir,dali_info[i].item(0),".txt"),"w") as f:
                    for item in words:
                        f.write("%s\n" % item)
        return True

    def get_single_lyric(self,entry):
        words = ["" for i in range(self.max_length)]
        my_annot = entry.annotations["annot"]["words"]
        length = len(my_annot[0:])
        for j in range(length):
            time1 = int(my_annot[0:][j].get("time")[0] * 44100)
            time2 = int(my_annot[0:][j].get("time")[1] * 44100)
            words[time1:time2] = [my_annot[0:][j].get("text")] * (time2 - time1)
        return words

    def json_download(self):
        dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
        dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")

        a = []
        for i in range(1, self.dataset_size+1):
            a.append(dali_info[i].item(0))
            entry = dali_data[dali_info[i].item(0)]
            path_save = self.json_dir
            name = dali_info[i].item(0)

            if (len(self.find_files(dali_info[i].item(0) + ".wav",self.audio_dir,))!= 0):
                entry.write_json(path_save, name)
        pass

    def txt_to_numpy(self, file_name):
        return np.loadtxt(file_name, delimiter="\n")

if(__name__=="__main__"):
    p = Preprocess(44100)
    print(p.find_max_length())
