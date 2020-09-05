#!/bin/bash

#set -x

python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_baseline_0.5_0.6.json --save_folder output/ --name coco_baseline_0_5_0_6 --num_workers 15

# This is in this long format to remove id 4 which is too slow
corruptions=(0 1 2 3 5 6 7 8 9 10 11 12 13 14)

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing baseline for c-${corruption_num} s-${severity_i}"
        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_baseline_c${corruption_num}s${severity_i}_0.5_0.6.json --save_folder output/ --name coco_baseline_c${corruption_num}s${severity_i}_0_5_0_6 --num_workers 15
    done
done

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing mcdrop25 for c-${corruption_num} s-${severity_i}"
        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop25_10_c${corruption_num}s${severity_i}_0.5_0.6.json --save_folder output/ --name coco_mcdrop25_10_c${corruption_num}s${severity_i}_0_5_0_6 --num_workers 15
    done
done

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing mcdrop25_no_retrain for c-${corruption_num} s-${severity_i}"
        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop25_10_no_retrain_c${corruption_num}s${severity_i}_0.5_0.6.json --save_folder output/ --name coco_mcdrop25_10_no_retrain_c${corruption_num}s${severity_i}_0_5_0_6 --num_workers 15
    done
done

