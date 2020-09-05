#!/bin/bash

#set -x

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name coco_baseline

# This is in this long format to remove id 4 which is too slow
corruptions=(0 1 2 3 5 6 7 8 9 10 11 12 13 14)

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing baseline for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name coco_baseline --corruption_num ${corruption_num} --severity ${severity_i}
    done
done

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing mcdrop25 for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_mcdrop25.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10 --corruption_num ${corruption_num} --severity ${severity_i}
    done
done

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing mcdrop25_no_retrain for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10_no_retrain --corruption_num ${corruption_num} --severity ${severity_i}
    done
done