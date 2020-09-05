#!/bin/bash

#set -x

python test.py --data data/coco2017.data --img-size 416 --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name coco_ensemble_5 --ensemble_main_name best_coco2017_ensemble_round2 --num_samples 5

python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_ensemble_5_0.5_0.6.json --save_folder output/ --name coco_ensemble_5_0_5_0_6 --num_workers 15

# This is in this long format to remove id 4 which is too slow
corruptions=(0 1 2 3 5 6 7 8 9 10 11 12 13 14)

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing ensemble 0.5 for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name coco_ensemble_5 --ensemble_main_name best_coco2017_ensemble_round2 --num_samples 5 --corruption_num ${corruption_num} --severity ${severity_i}

        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_ensemble_5_c${corruption_num}s${severity_i}_0.5_0.6.json --save_folder output/ --name coco_ensemble_5_c${corruption_num}s${severity_i}_0_5_0_6 --num_workers 15
    done
done

echo "Dont forget to close the script command"