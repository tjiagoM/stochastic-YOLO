'''
This script is to solve the issue with the COCO images folder that has ~40k images even though the validation list is only ~5k images. This affects evaluation with the PDQ script.
'''

from pathlib import Path
import os
import json

path_correct_list = '../coco/5k.txt'

path_correct_list = str(Path(path_correct_list))  # os-agnostic
parent = str(Path(path_correct_list).parent) + os.sep
# Getting the 5k img IDs

with open(path_correct_list, 'r') as f:
    f = f.read().splitlines()
    f = [x.replace('./', parent) if x.startswith('./') else x for x in f]  # local to global path

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.dng']
img_files = [x.replace('/', os.sep) for x in f if os.path.splitext(x)[-1].lower() in img_formats]

img_ids = [int(Path(x).stem.split('_')[-1]) for x in img_files]


# Removing those images from coco2014 ground truth
filename = '../coco/annotations/instances_val2014.json'
with open(filename, 'r') as f:
    data = json.load(f)

data['images'] = [elem for elem in data['images'] if elem['id'] in img_ids ]
data['annotations'] = [elem for elem in data['annotations'] if elem['image_id'] in img_ids ]
    
# Saving new json
with open('filtered_instances_val2014.json', 'w') as f:
    json.dump(data, f, indent=4)