#!/bin/bash

python train.py --data data/coco2017.data --epochs 200 --batch-size 40 --name coco2017_ensemble_round2_1 --weights weights/last_coco2017_ensemble1.pt --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 200 --batch-size 40 --name coco2017_ensemble_round2_2 --weights weights/last_coco2017_ensemble2.pt --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 200 --batch-size 40 --name coco2017_ensemble_round2_3 --weights weights/last_coco2017_ensemble3.pt --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 200 --batch-size 40 --name coco2017_ensemble_round2_4 --weights weights/last_coco2017_ensemble4.pt --cfg cfg/yolov3-custom.cfg --img-size 416
python train.py --data data/coco2017.data --epochs 200 --batch-size 40 --name coco2017_ensemble_round2_5 --weights weights/last_coco2017_ensemble5.pt --cfg cfg/yolov3-custom.cfg --img-size 416
