#!/bin/bash

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop50.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop50_10_no_retrain --device='cpu'

python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop50_10_no_retrain_0.5_0.6.json --save_folder output/ --name coco_mcdrop50_10_no_retrain_0_5_0_6 --num_workers 15

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop75_10_no_retrain --device='cpu'

python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco_mcdrop75_10_no_retrain_0.5_0.6.json --save_folder output/ --name coco_mcdrop75_10_no_retrain_0_5_0_6 --num_workers 15