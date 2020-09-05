#!/bin/bash

#set -x

#
# Making 2 separate for cycles to be able to parallelise earlier other scripts also using the GPUs

drop_rates=(0.01 0.05 0.1 0.15 0.2)
drop_names=('01' '05' '10' '15' '20')

for i in {0..4}
do
    echo "testing mcdrop${drop_names[i]}_no_retrain for no corruption"
    python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop${drop_names[i]}_10_no_retrain --new_drop_rate ${drop_rates[i]} --batch-size 20

    # This is in this long format to remove id 4 which is too slow
    corruptions=(0 1 2 3 5 6 7 8 9 10 11 12 13 14)

    for corruption_num in ${corruptions[@]} ; do
        for severity_i in {1..5}
        do
            echo "testing mcdrop${drop_names[i]}_no_retrain for c-${corruption_num} s-${severity_i}"
            python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop${drop_names[i]}_10_no_retrain --corruption_num ${corruption_num} --severity ${severity_i}  --new_drop_rate ${drop_rates[i]} --batch-size 20
        done
    done
done



for i in {0..4}
do
    echo "evaluating mcdrop${drop_names[i]}_no_retrain for no corruption"
    python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop${drop_names[i]}_10_no_retrain_0.5_0.6.json --save_folder output/ --name coco_mcdrop${drop_names[i]}_10_no_retrain_0_5_0_6 --num_workers 15


    for corruption_num in ${corruptions[@]} ; do
        for severity_i in {1..5}
        do
            echo "evaluating mcdrop${drop_names[i]}_no_retrain for c-${corruption_num} s-${severity_i}"
            python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop${drop_names[i]}_10_no_retrain_c${corruption_num}s${severity_i}_0.5_0.6.json --save_folder output/ --name coco_mcdrop${drop_names[i]}_10_no_retrain_c${corruption_num}s${severity_i}_0_5_0_6 --num_workers 15
        done
    done
done

echo "Dont forget to close the script command"