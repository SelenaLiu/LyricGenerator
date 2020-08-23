import multiprocessing as mp
import numpy as np
import DALI as dali_code
import pickle, os, subprocess, librosa

class Preprocess():

    def __init__(self,sr,batch_size):
        self.converted_audio = "conv_aud.p"
        self.saved_audio_np = "saved_np.p"
        self.lyrics_np = "lyrics.p"
        self.audio_np = "audio.p"

        self.dataset_size = 5358
        self.batch_size = batch_size
        self.sample_rate = sr
        self.max_duration = 330
        self.max_length = self.sample_rate*self.max_duration # Changed from 0

        self.data_dir = "/mnt/d/Repositories/LyricGenerator/data"
        self.audio_dir = os.path.join(self.data_dir,"Audio")
        self.lyric_dir = os.path.join(self.data_dir,"Lyrics")
        self.json_dir = os.path.join(self.data_dir,"JSON")
        self.dali_dir = os.path.join(self.data_dir,"DALI_v1.0")
        self.extra_dir = os.path.join(self.data_dir,"Extra Audio")

        self.audio_np_dir = os.path.join(self.data_dir,"audio_np")
        self.lyrics_np_dir = os.path.join(self.data_dir,"lyrics_np")

        print("[dali] caching dataset")
        self.dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
        self.dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")
        print("cache complete")
        print(self.dali_info)
        print()
        print(self.dali_data)

    def get_dali_data(self):
        return self.dali_data

    def load_audio(self,file_path,target_sr):
        try:
            signal, sample_rate = librosa.load(file_path,sr=target_sr,mono=True)
            samples = librosa.resample(signal,sample_rate,target_sr)
            return samples,sample_rate
        except:
            print("Error Loading"+file_path)
            return -1,-1
        pass

    def change_ext(self,filename,out):
        return os.path.splitext(filename)[0]+out

    def find_max_length(self):
        max_duration = 0
        for file in os.listdir(self.audio_dir):
            if file.endswith(".wav"):
                duration = librosa.get_duration(filename=os.path.join(self.audio_dir,file),sr=self.sample_rate)
                if(duration>330):
                    os.rename(os.path.join(self.audio_dir,file), os.path.join(self.extra_dir ,file))
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

    def compile_audio(self):
        all_audio_np = []
        save = []

        count = 1
        batch = 1

        for i in range(1, self.dataset_size+1):
            entry = self.dali_data[self.dali_info[i].item(0)]
            filename = self.dali_info[i].item(0) + ".wav"
            if (len(self.find_files(filename,self.audio_dir))!= 0):
                print("[audio] Adding "+str(self.dali_info[i].item(0))+" to batch "+str(batch))
                signal,sr = self.load_audio(os.path.join(self.audio_dir,filename),self.sample_rate)
                if(sr!=-1):
                    os.remove(os.path.join(self.audio_dir,filename))
                    save.append(signal)
                    count = count + 1
            if count % self.batch_size == 0:
                print("[audio] Saving batch "+str(batch))
                all_audio_np.append(save)
                pickle_name = str(batch)+self.audio_np
                pickle.dump(save,open(os.path.join(self.audio_np_dir,pickle_name),"wb"))
                batch = batch + 1
                save = []
        all_audio_np.append(save)
        pickle_name = str(batch)+self.audio_np
        pickle.dump(save,open(os.path.join(self.audio_np_dir,pickle_name),"wb"))
        return np.array(all_audio_np)

    def compile_lyrics(self):
        all_lyrics_np = []
        save = []

        count = 1
        batch = 1

        for i in range(1, self.dataset_size+1):
            entry = self.dali_data[self.dali_info[i].item(0)]
            filename = self.dali_info[i].item(0) + ".wav"
            if (len(filename,self.find_files(filename,self.audio_dir))!= 0):
                print("[lyrics] Adding "+str(self.dali_info[i].item(0))+" to batch "+str(batch))
                words = get_single_lyric(entry)
                words_np = np.array(words)
                save.append(words_np)
                count = count + 1
            if count % self.batch_size == 0:
                print("[lyrics] Saving batch "+str(batch))
                all_lyrics_np.append(save)
                pickle_name = str(batch)+self.lyrics_np
                pickle.dump(save,open(os.path.join(self.lyrics_np_dir,pickle_name),"wb"))
                batch = batch + 1
                save = []
        all_lyrics_np.append(save)
        pickle_name = str(batch)+self.lyrics_np
        pickle.dump(save,open(os.path.join(self.lyrics_np_dir,pickle_name),"wb"))
        return np.array(all_lyrics_np)

    def compile_audio_and_lyrics_mp(self):#Not Functional
        all_lyrics_np = []
        all_audio_np = []
        save_lyrics_np = []
        save_audio_np = []
        data_q = []

        count = 1
        batch = 1

        for i in range(1, self.dataset_size+1):
            item = self.dali_info[i].item(0)
            entry = self.dali_data[item]
            filename = item + ".wav"
            if(len(filename,self.find_files(filename,self.audio_dir))!= 0):
                data_q.append(item)
                count = count + 1
            if(count == self.batch_size):
                #pool = mp.Pool()
                #pool.map()
                pass
        all_lyrics_np.append(save)
        pickle_name = str(batch)+self.lyrics_np
        pickle.dump(save,open(os.path.join(self.lyrics_np_dir,pickle_name),"wb"))
        return np.array(all_lyrics_np)

    def dali_json_to_txt(self):
        for i in range(1, self.dataset_size+1):
            entry = self.dali_data[self.dali_info[i].item(0)]
            if (len(self.find_files(self.dali_info[i].item(0) + ".wav",self.audio_dir))!= 0):
                # print(entry.annotations["annot"].keys())
                words = get_single_lyric(entry)
                with open(os.path.join(self.lyric_dir,self.dali_info[i].item(0),".txt"),"w") as f:
                    for item in words:
                        f.write("%s\n" % item)
        return True

    def append_to_batch(self,item,save_lyrics_np,save_audio_np):
        signal,sr = self.load_audio(os.path.join(self.audio_dir,filename),self.sample_rate)
        if(sr!=-1):
            os.remove(os.path.join(self.audio_dir,filename))
            save.append(signal)
            count = count + 1

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
        a = []
        for i in range(1, self.dataset_size+1):
            a.append(self.dali_info[i].item(0))
            entry = self.dali_data[self.dali_info[i].item(0)]
            path_save = self.json_dir
            name = self.dali_info[i].item(0)

            if (len(self.find_files(self.dali_info[i].item(0) + ".wav",self.audio_dir,))!= 0):
                entry.write_json(path_save, name)
        pass

    def audio_download(self):
        path_audio = os.path.join(data_dir,"Audio")
        errors = dali_code.get_audio(self.dali_info, path_audio, skip=[], keep=[])
        pass

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
                    os.remove(os.path.join(self.audio_dir,file))
                else:
                    print("Converting "+file+" from mp3 to wav")
                    subprocess.call(["ffmpeg","-i",(os.path.join(self.audio_dir,file)),(os.path.join(self.audio_dir,wav_filename))])
                converted.append(file)
        pickle.dump(converted,open(self.converted_audio,"wb"))
        return converted

    def find_files(self, filename, search_path):
        result = []
        # Walking top-down from the root
        for root, dir, files in os.walk(search_path):
            if filename in files:
                result.append(os.path.join(root, filename))
        return result

    def txt_to_numpy(self, file_name):
        return np.loadtxt(file_name, delimiter="\n")

if(__name__=="__main__"):
    p = Preprocess(44100,128)
    #p.compile_audio()
    #p.compile_lyrics()
