import multiprocessing as mp
import numpy as np
import DALI as dali_code
import pickle, os, subprocess, librosa
import collections

def find_files(filename, search_path):
    result = []
    # Walking top-down from the root
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result

def load_audio(file_path,target_sr):
    try:
        signal, sample_rate = librosa.load(file_path,sr=target_sr,mono=True)
        samples = librosa.resample(signal,sample_rate,target_sr)
        return samples,sample_rate
    except:
        print("Error Loading"+file_path)
        return -1,-1
    pass

def save_single_audio_lyric(data):
    #print(data[0][0])
    #print("______________________________________")
    print(data[2][1])

    id = data[0][0]
    print(type(id))
    lyric_np = str(id)+"_lyr.p"
    audio_np = str(id)+"_aud.p"
    filename = id+".wav"
    if(len(find_files(filename,data[2][1]))!= 0):
        print("["+id[0:5]+"..."+id[-5:0]+"] loading audio and lyrics")
        audio_np_single,sr = load_audio(os.path.join(data[2][1],filename),data[2][0])
        if(sr!=-1):
            os.remove(os.path.join(data[2][1],filename))
            entry = data[1][int(id)]
            lyric = get_single_lyric(entry)
            lyric_np_single = np.array(words)
            print("["+id[0:5]+"..."+id[-5:0]+"] pickling audio and lyrics")
            pickle.dump(audio_np_single,open(os.path.join(data[2][2],audio_np),"wb"))
            pickle.dump(lyric_np_single,open(os.path.join(data[2][3],lyric_np),"wb"))

class Preprocess():

    def __init__(self,sr,batch_size):
        self.converted_audio = "conv_aud.p"
        self.saved_audio_np = "saved_np.p"
        self.lyrics_np = "lyrics.p"
        self.audio_np = "audio.p"
        self.dali_data_p = "dali_data.p"
        self.dali_info_p = "dali_info.p"

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
        if(os.path.exists(self.dali_data_p)):
            print("[dali_data] reading from file...")
            self.dali_data = pickle.load(open(self.dali_data_p,"rb"))
            print("[dali_data] read complete")
        else:
            self.dali_data = dali_code.get_the_DALI_dataset(self.dali_dir)
            pickle.dump(self.dali_data,open(self.dali_data_p,"wb"))
        if(os.path.exists(self.dali_info_p)):
            print("[dali_info] reading from file...")
            self.dali_info = pickle.load(open(self.dali_info_p,"rb"))
            print("[dali_info] read complete")
        else:
            self.dali_info = dali_code.get_info(self.dali_dir + "/info/DALI_DATA_INFO.gz")
            pickle.dump(self.dali_info,open(self.dali_info_p,"wb"))
        print("cache complete")
        pass

    def get_dali_data(self):
        return self.dali_data

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
            if (len(find_files(filename,self.audio_dir))!= 0):
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
            if (len(find_files(filename,self.audio_dir))!= 0):
                print("[lyrics] Adding "+str(self.dali_info[i].item(0))+" to batch "+str(batch))
                words = self.get_single_lyric(entry)
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

        data = [list(a) for a in zip(np.sort(self.dali_info,axis=0)[1:],self.dali_data,[[self.sample_rate,self.audio_dir,self.audio_np_dir,self.lyrics_np_dir] for i in range(self.dataset_size)])]

        #print(data[0][0])
        #print(data[0][1])
        #print(type(data[0][0]))
        #print(type(data[0][1]))
        print(np.sort(self.dali_info,axis=0))
        print()
        od = collections.OrderedDict(sorted(d.items()))
        print(self.dali_data.keys())
        #entry = self.dali_data[item]
        #pool = mp.Pool()
        #pool.map(save_single_audio_lyric,data)
        #pool.close()
        #pool.join()
        pass

    def dali_json_to_txt(self):
        for i in range(1, self.dataset_size+1):
            entry = self.dali_data[self.dali_info[i].item(0)]
            if (len(find_files(self.dali_info[i].item(0) + ".wav",self.audio_dir))!= 0):
                # print(entry.annotations["annot"].keys())
                words = get_single_lyric(entry)
                with open(os.path.join(self.lyric_dir,self.dali_info[i].item(0),".txt"),"w") as f:
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
        a = []
        for i in range(1, self.dataset_size+1):
            a.append(self.dali_info[i].item(0))
            entry = self.dali_data[self.dali_info[i].item(0)]
            path_save = self.json_dir
            name = self.dali_info[i].item(0)

            if (len(find_files(self.dali_info[i].item(0) + ".wav",self.audio_dir,))!= 0):
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
    p.compile_audio_and_lyrics_mp()
