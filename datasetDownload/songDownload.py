import DALI as dali_code
import numpy as np

dali_data_path = "C:\\Users\\Himanish Jindal\\Desktop\\HackThe6ix\\DALI_v1.0"
dali_data = dali_code.get_the_DALI_dataset(dali_data_path)
dali_info = dali_code.get_info(dali_data_path + "\\info\\DALI_DATA_INFO.gz")

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

path_audio = "C:\\Users\\Himanish Jindal\\Desktop\\HackThe6ix\\Audio Files"
errors = dali_code.get_audio(dali_info, path_audio, skip=[], keep=[])
# errors -> ['dali_id', 'youtube_url', 'error']
print("DONE")

