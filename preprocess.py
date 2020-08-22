from utils import Utils
import numpy as np

class Preprocess():

    def __init__():
        self.ut = Utils()

    def process_data(self, list_songs: List):
        #gets longest song from entire array
        list_songs.sort(key=len)
        max_length = list_songs[-1].shape[0]
        for element in list_songs:
            element_len = element.shape[0]
            N = max_length - element_len
            np.pad(element, (0, N), 'constant')


    # def equalize_length():
    #     #needs code
    #     #pads all songs at end to be same size as max length
    #     pass
