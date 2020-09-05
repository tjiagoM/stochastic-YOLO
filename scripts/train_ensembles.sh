#!/bin/bash

#python train.py --data data/coco2017.data --epochs 100 --batch-size 40 --name coco2017_ensemble_1 --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 100 --batch-size 40 --name coco2017_ensemble_2 --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 100 --batch-size 40 --name coco2017_ensemble_3 --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 100 --batch-size 40 --name coco2017_ensemble_4 --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 100 --batch-size 40 --name coco2017_ensemble_5 --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
