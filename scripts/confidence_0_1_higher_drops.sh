#!/bin/bash

#set -x

#
# Making 2 separate for cycles to be able to parallelise earlier other scripts also using the GPUs

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop50.cfg --conf-thres=0.1 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop50_10_no_retrain --device=1

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_mcdrop75.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.1 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name coco_mcdrop75_10  --device=1

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.1 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop75_10_no_retrain  --device=1

# This is in this long format to remove id 4 which is too slow
corruptions=(0 1 2 3 5 6 7 8 9 10 11 12 13 14)

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "testing mcdrop50_no_retrain for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop50.cfg --conf-thres=0.1 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop50_10_no_retrain --corruption_num ${corruption_num} --severity ${severity_i}  --device=1

        echo "testing mcdrop75 for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_mcdrop75.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.1 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name coco_mcdrop75_10 --corruption_num ${corruption_num} --severity ${severity_i}  --device=1

        echo "testing mcdrop75_no_retrain for c-${corruption_num} s-${severity_i}"
        python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.1 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop75_10_no_retrain --corruption_num ${corruption_num} --severity ${severity_i}  --device=1
    done
done

echo "evaluating mcdrop50_no_retrain for no corruption"
python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop50_10_no_retrain_0.1_0.6.json --save_folder output/ --name coco_mcdrop50_10_no_retrain_0_1_0_6 --num_workers 10

echo "evaluating mcdrop75 for no corruption"
python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop75_10_0.1_0.6.json --save_folder output/ --name coco_mcdrop75_10_0_1_0_6 --num_workers 10

echo "evaluating mcdrop75_no_retrain for no corruption"
python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop75_10_no_retrain_0.1_0.6.json --save_folder output/ --name coco_mcdrop75_10_no_retrain_0_1_0_6 --num_workers 10

for corruption_num in ${corruptions[@]} ; do
    for severity_i in {1..5}
    do
        echo "evaluating mcdrop50_no_retrain for c-${corruption_num} s-${severity_i}"
        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop50_10_no_retrain_c${corruption_num}s${severity_i}_0.1_0.6.json --save_folder output/ --name coco_mcdrop50_10_no_retrain_c${corruption_num}s${severity_i}_0_1_0_6 --num_workers 10

        echo "evaluating mcdrop75 for c-${corruption_num} s-${severity_i}"
        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop75_10_c${corruption_num}s${severity_i}_0.1_0.6.json --save_folder output/ --name coco_mcdrop75_10_c${corruption_num}s${severity_i}_0_1_0_6 --num_workers 10

        echo "evaluating mcdrop75_no_retrain for c-${corruption_num} s-${severity_i}"
        python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop75_10_no_retrain_c${corruption_num}s${severity_i}_0.1_0.6.json --save_folder output/ --name coco_mcdrop75_10_no_retrain_c${corruption_num}s${severity_i}_0_1_0_6 --num_workers 10
    done
done

echo "Dont forget to close the script command"