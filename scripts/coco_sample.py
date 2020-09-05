'''
Script to create a smaller COCO dataset (train or validation) according to some percentage which will be filtered out from each category, instead of just completely random overall.
'''
import sys
sys.path.append('./cocoapi/PythonAPI/')

import os
from pycocotools.coco import COCO
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--instance', type=str, default='train2017', help='which instance to sample from (eg train2017/val2017)')
parser.add_argument('--perc', type=float, default=0.05, help='how much to sample from each category')
parser.add_argument('--min_number', type=int, default=500, help='how many images there should exist at least, regardless of --perc')

opt = parser.parse_args()

instance_name = opt.instance
perc = opt.perc
min_number = opt.min_number

print(f'Sampling {perc}% or {min_number} for {instance_name}')

coco = COCO(f'../coco/annotations/instances_{instance_name}.json')

all_ids = set()

for cat_id in coco.getCatIds():
    print(f'For category {cat_id}:')
    img_ids = coco.getImgIds(catIds=[cat_id])
    print(f'-- {len(img_ids)} images')
    
    # Keeping perc% of original images with that category, or at least min_number
    if len(img_ids) > min_number:
        perc_num = int(perc * len(img_ids))
        num_filter = perc_num if perc_num > min_number else min_number

        img_ids = random.sample(img_ids, num_filter)
    print(f'-- {len(img_ids)} sampled images')
    
    all_ids.update(img_ids)
    
print(f'Total new size: {len(all_ids)} images')

print('Saving new list...')

with open(f'../coco/{instance_name}_sampled.txt', 'w')  as f:
    for elem in all_ids:
        f.write(os.path.join('.', 'images', instance_name,'{:012}'.format(img_ids[0]) + '.jpg') + '\n')