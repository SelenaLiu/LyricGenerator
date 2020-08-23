import DALI as dali_code
import numpy as np
import os

data_dir = "/mnt/d/Repositories/LyricGenerator/data"
dali_data_path = os.path.join(data_dir,"DALI_v1.0")
dali_data = dali_code.get_the_DALI_dataset(dali_data_path)
dali_info = dali_code.get_info(os.path.join(dali_data_path,"info/DALI_DATA_INFO.gz"))
print(dali_info)
"""
a = []

for i in range(1, 950):
    a.append(dali_info[i].item(0))

# print(a)

entry = dali_data["001940b614eb43f4a0c826d49a67d66d"]
# type(entry)  # -> DALI.Annotations.Annotations

path_save = "C:\\Users\\Himanish Jindal\\Desktop\\HackThe6ix\\JSON files"
name = "first song"
# export
entry.write_json(path_save, name)
# import
# my_json_entry = dali_code.Annotations()
# my_json_entry.read_json(os.path.join(path_save, name + ".json"))
print("Done")
"""

path_audio = os.path.join(data_dir,"Audio")
errors = dali_code.get_audio(dali_info, path_audio, skip=[], keep=[])
# errors -> ['dali_id', 'youtube_url', 'error']
print("DONE")
