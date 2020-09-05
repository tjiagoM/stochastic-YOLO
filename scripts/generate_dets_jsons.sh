#!/bin/bash

echo "coco_baseline 0.5 0.6"

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name coco_baseline

echo "coco_mcdrop25_10_no_retrain 0.5"

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10_no_retrain

echo "coco_mcdrop25_10 0.5"
python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_mcdrop25.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10

#echo "coco_mcdrop25_10_no_retrain 0.1"

#python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.1 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10_no_retrain

#echo "coco_mcdrop25_10 0.1"
#python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_mcdrop25.pt --cfg cfg/yolov3-mcdropout25.cfg --conf-thres=0.1 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10


