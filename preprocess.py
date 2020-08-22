from utils import Utils
import numpy as np
import DALI as dali_code

import pickle, os, subprocess


class Preprocess():

    def __init__(self,sr):
        self.ut = Utils()

        self.converted_audio = "conv_aud.p"
        self.saved_audio_np = "saved_np.p"

        self.dataset_size = 5358
        self.sample_rate = sr
        self.max_length = 0

        self.data_dir = "/mnt/d/Repositories/LyricGenerator/data"
        self.audio_dir = os.path.join(data_dir,"Audio")
        self.lyric_dir = os.path.join(data_dir,"Lyrics")
        self.json_dir = os.path.join(data_dir,"JSON")
        self.dali_dir = os.path.join(data_dir,"DALI_v1.0")

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
                wav_filename = self.ut.change_ext(file,".wav")
                if(os.path.isfile(wav_filename)):
                    print("Skipping "+wav_filename+", already converted.")
                    os.remove(file)
                else:
                    print("Converting "+file+" from mp3 to wav")
                    subprocess.call(["ffmpeg","-i",(os.path.join(audio_path,file)),(os.path.join(audio_path,wav_filename))])
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

    def find_files(self, filename, search_path):
        result = []
        # Walking top-down from the root
        for root, dir, files in os.walk(search_path):
            if filename in files:
                result.append(os.path.join(root, filename))
        return result

    def jsontotxt(self):
        dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
        dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")

        for i in range(1, self.dataset_size+1):
            entry = dali_data[dali_info[i].item(0)]
            if (
                len(
                    self.find_files(
                        dali_info[i].item(0) + ".json",
                        self.json_dir,
                    )
                )
                != 0
            ):
                # print(entry.annotations["annot"].keys())
                max_length = self.max_length()
                words = ["" for i in range(max_length)]
                my_annot = entry.annotations["annot"]["words"]
                length = len(my_annot[0:])
                for j in range(length):
                    time1 = int(my_annot[0:][j].get("time")[0] * 44100)
                    time2 = int(my_annot[0:][j].get("time")[1] * 44100)
                    print(time1, time2)
                    # for k in range((time1*44100, (time2*44100)):
                    words[time1:time2] = [my_annot[0:][j].get("text")] * (time2 - time1)
                with open(os.path.join(self.lyric_dir,dali_info[i].item(0),".txt"),"w") as f:
                    for item in words:
                        f.write("%s\n" % item)
        pass

    def json_download(self):
        dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
        dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")

        a = []
        for i in range(1, self.dataset_size+1):
            a.append(dali_info[i].item(0))
            entry = dali_data[dali_info[i].item(0)]
            path_save = self.json_dir
            name = dali_info[i].item(0)

            if (
                len(
                    self.find_files(
                        dali_info[i].item(0) + ".wav",
                        self.audio_dir,
                    )
                )
                != 0
            ):
                entry.write_json(path_save, name)
        pass

    # def equalize_length():
    #     #needs code
    #     #pads all songs at end to be same size as max length
    #     pass
