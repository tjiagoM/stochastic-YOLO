# Probabilistic OD

## Repository preliminaries

This repository was originally forked from https://github.com/ultralytics/yolov3

### External packages

For this repo it was decided to not install pycocotools for a better (and tracked) flexibility when editing some classes. Folder `cocoapi` is then a fork from the original [repository](https://github.com/cocodataset/cocoapi), installed locally in the form of a git submodule. To import it, it is only necessary to include the folder in the system path:

```python
import sys
sys.path.append('./cocoapi/PythonAPI/')
```

We also forked an external repository for calculation of the PDQ score, but included as submodule in this repo for flexibility. Instructions there: https://github.com/david2611/pdq_evaluation (originally for the Robotic Vision Challenge 1 and adapted for COCO dataset).

### Repository structure

This repository follows the structure of the original repository, with some changes:

 * `cgf`: Where all the (Darknet) configuration files, all from the original repository. For this repository in specific, `yolov3-custom.cfg` and `yolov3-mcdropout.cfg` are the ones added.
 * `cocoapi` and `pdq_evaluation`: git submodules with the external packages as defined in previous section
 * `data`: configuration files for information about the data used in the scripts. For this repository, we are mostly using `coco2017.data` for the COCO 2017 dataset. We included an extra key named `instances_path` which is not present in the `.data` files from the original repository. This allowed for the use of both COCO 2014 and 2017 in the same project.
 * `output`: Mostly temporary output files from different runs.
 * `results`: The CSVs with the metrics calculated from `pdq_evaluation/evaluate.py`. Decision of having one CSV in separate for each model and corruption/severity due to flexibility in calling parallel evaluation scripts.
 * `scripts`: some bash scripts to run over the terminal for several python tasks
 * `weights`: Folder with trained models


## Overall commands

### Training from scratch on COCO 2017

To train from scratch yolov3, with image size input of 416x416 (defined in yolov3-custom.cfg), including scaling of images in the train/test set for 100 epochs:

```bash
python train.py --data data/coco2017.data --epochs 100 --batch-size 20 --name coco2017_scratch --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
```

After these 100 epochs, and given we passed a `--name` flag, there will be a file in the root repository with the evolution of the training metrics named `results_coco2017_scratch.txt`. We can plot training evolution with `python -c "from utils import utils; utils.plot_results(name_to_plot='coco2017_scratch')"`:

![coco2017 training](output/results_coco2017_scratch.png "COCO 2017 training")

These plots don't seem to agree with the metrics calculated at inference time (eg. mAP and F1 are much lower in the training plots, even though supposedly the same testing script is used both in train/test times).

We further trained the model for another 100 epochs:
```bash
python train.py --data data/coco2017.data --epochs 200 --batch-size 20 --name coco2017_scratch_round2 --weights weights/last_coco2017_scratch.pt --cfg cfg/yolov3-custom.cfg --img-size 416
```


### Evaluating previous model

**With confidence threshold of 0.001 and IoU threshold of 0.6 for NMS:**

```bash
python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.001 --iou-thres=0.6 --name coco2017_scratch_round2
```

The previous command will generate a json file in `output/dets_coco2017_scratch_round2_0.001_0.6.json` in a format that can be used by the official MC COCO Evaluation API (in `cocoapi/`). The evaluation metrics are:

```
Class    Images   Targets         P         R   mAP@0.5        F1
  all     5e+03  3.63e+04     0.499     0.601     0.546     0.542
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.681
```

**With confidence threshold of 0.1 and IoU threshold of 0.6 for NMS:**

```bash
python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.1 --iou-thres=0.6 --name coco2017_scratch_round2
```

The json file created this time will be `output/dets_coco2017_scratch_0.1_0.6.json` instead, with the following metrics:

```
Class    Images   Targets         P         R   mAP@0.5        F1
  all     5e+03  3.63e+04       0.5     0.601     0.507     0.543
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
```


### Further evaluations (PDQ and LRP)

As explained in `pdq_evaluation/`, it is necessary to first convert the json from results to another json format; however, this is done automatically inside `test.py`, in which the corresponding json will have a `_converted` in the name (more details below).


And to evalute (`evaluate.py` script took more than 30 minutes to run in our servers for 0.001 confidence threshold):

```bash
python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco2017_scratch_round2_0.001_0.6.json --save_folder output/ --set_cov 0 --name baseline_0_001__0_6 | tee tmp_output.txt
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.391
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.175
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.518
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.300
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.681
 
PDQ: 0.008719
mAP: 0.365565
avg_pPDQ:0.153878
avg_spatial:0.121277
avg_label:0.403718
avg_foreground:0.544703
avg_background:0.308979
TP:29059
FP:476040
FN:7722
moLRP:0.695760
moLRPLoc:0.167382
moLRPFP:0.289060
moLRPFN:0.475230
```

And for 0.1 confidence threshold, much quicker though:

```bash
python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco2017_scratch_round2_0.1_0.6.json --save_folder output/ --set_cov 0 --name baseline_0_1__0_6 | tee tmp_output.txt
```

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.512
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.151
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.479
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619


PDQ: 0.071953
mAP: 0.344670
avg_pPDQ:0.199604
avg_spatial:0.145329
avg_label:0.552651
avg_foreground:0.652270
avg_background:0.280292
TP:21844
FP:23816
FN:14937
moLRP:0.696304
moLRPLoc:0.167165
moLRPFP:0.281327
moLRPFN:0.477503
```

### After round 2, with confidence threshold of 0.1

python train.py --data data/coco2017.data --epochs 100 --batch-size 20 --name coco2017_mcdropout --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdropout.cfg --img-size 416 --dropout_ids 82 95 108

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop25 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110


python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.1 --iou-thres=0.6 --name coco2017_scratch_round2

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.1 --iou-thres=0.6 --name coco2017_scratch

![coco2017 training round2](output/results_coco2017_scratch_round2.png "COCO 2017 training round 2")


### How MCDropout was trained, and evaluation was done

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop25 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop50 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop50.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop75 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name coco_mcdrop25_10_no_retrain


### Ensembles part
bash scripts/ensembles_0_5.sh


### Plotting results

```bash
$ bash scripts/plot_corruption_results.sh
```

```bash
$ python -c "from utils import utils; utils.generate_paper_plots()"
```

### LaTeX outputs

python -c "from utils import utils; utils.generate_latex_table(['mAP', 'PDQ', 'avg_label', 'avg_spatial'])" | tee latex_output.txt


### Getting Unknowns

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name unknown_tests --get_unknowns --batch-size 40

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_mcdrop25.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name unknown_testsMCD25 --get_unknowns --batch-size 20

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name unknown_testsMCD25NO --get_unknowns --batch-size 20


python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name unknown0_1 --get_unknowns --batch-size 40

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name unknown0_1MCD25NO --get_unknowns --batch-size 20


python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.8 --iou-thres=0.6 --name unknown0_8 --get_unknowns --batch-size 40

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.8 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name unknown0_8MCD25NO --get_unknowns --batch-size 20


python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name taco --batch-size 40 --only_inference

python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.8 --iou-thres=0.6 --name taco --batch-size 40  --get_unknowns --only_inference

python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.8 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name tacoMCD25 --get_unknowns --batch-size 40 --only_inference

python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.8 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name tacoMCD75 --get_unknowns --batch-size 40 --only_inference

python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_mcdrop75.pt --cfg cfg/yolov3-mcdrop75.cfg --conf-thres=0.8 --iou-thres=0.6 --dropout_at_inference --num_samples 10 --name tacoMCD75-X --get_unknowns --batch-size 40 --only_inference