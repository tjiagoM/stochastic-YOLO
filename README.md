![license](https://img.shields.io/badge/license-Apache%202.0-blue)
[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-b31b1b.svg)](https://arxiv.org/abs/0000.00000)
# Stochastic-YOLO
*Tiago Azevedo, René de Jong, Partha Maji*

This repository contains all the code necessary to run and further extend the experiments presented in the following ArXiv preprint: [https://arxiv.org/abs/0000.00000](https://arxiv.org/abs/0000.00000)

## Abstract


## Repository preliminaries

This repository was originally forked from https://github.com/ultralytics/yolov3

### External packages

For this repo it was decided to not install pycocotools in the python environment for a better (and tracked) flexibility when editing some classes. Folder `cocoapi` is then a fork from the original [repository](https://github.com/cocodataset/cocoapi), installed locally in the form of a git submodule. To import it, it is only necessary to include the folder in the system path whenever needed:

```python
import sys
sys.path.append('./cocoapi/PythonAPI/')
```

We also forked an [external repository](https://github.com/david2611/pdq_evaluation) for calculation of the PDQ score, but included as submodule in this repo for flexibility. Instructions there: https://github.com/tjiagoM/pdq_evaluation (originally for the Robotic Vision Challenge 1 and adapted for COCO dataset).

### Repository structure

This repository follows the structure of the original repository, with some changes:

 * `cgf`: Where all the (Darknet) configuration files, all from the original repository. For this repository in specific, `yolov3-custom.cfg` and `yolov3-mcdropXX.cfg` are the ones added.
 * `cocoapi` and `pdq_evaluation`: git submodules with the external packages as defined in previous section
 * `data`: configuration files for information about the data used in the different scripts. For this repository, we are mostly using `coco2017.data` for the COCO 2017 dataset. We included an extra key named `instances_path` which is not present in the `.data` files from the original Ultalytics' repository.
 * `output`: Mostly temporary output files from different runs.
 * `results`: The CSVs with the metrics calculated from `pdq_evaluation/evaluate.py`. Decision of having one CSV in separate for each model and corruption/severity was due to flexibility in calling parallel evaluation scripts.
 * `scripts`: some bash scripts to run over the terminal for several python tasks
 * `weights`: Folder with trained models


## Overall commands

### Training from scratch on COCO 2017

To train from scratch yolov3, with image size input of 416x416 (defined in yolov3-custom.cfg), including scaling of images in the train/test set for 100 epochs:

```bash
python train.py --data data/coco2017.data --epochs 100 --batch-size 20 --name coco2017_scratch --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
```

After these 100 epochs, and given we passed a `--name` flag, there will be a file in the root repository with the evolution of the training metrics named `results_coco2017_scratch.txt`. 

We further trained the model for another 100 epochs:
```bash
python train.py --data data/coco2017.data --epochs 200 --batch-size 20 --name coco2017_scratch_round2 --weights weights/last_coco2017_scratch.pt --cfg cfg/yolov3-custom.cfg --img-size 416
```


### Evaluating previous model

**With confidence threshold of 0.1 and IoU threshold of 0.6 for NMS:**

```bash
python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.1 --iou-thres=0.6 --name coco2017_scratch_round2
```
The previous command will generate a json file in `output/dets_coco2017_scratch_round2_0.1_0.6.json` in a format that can be used by the official MC COCO Evaluation API (in `cocoapi/`). Some of the evaluation metrics in the terminal output are:

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


### Further evaluations (PDQ and others)

As explained in `pdq_evaluation/`, it is necessary to first convert the json from results to another json format; however, this is done automatically inside `test.py`, in which the corresponding json will have a `_converted` in the name. To evaluate the model for 0.1 confidence threshold, run the following command:

```bash
python -u pdq_evaluation/evaluate.py --test_set 'coco' --gt_loc ../coco/annotations/instances_val2017.json --det_loc output/dets_converted_coco2017_scratch_round2_0.1_0.6.json --save_folder output/ --set_cov 0 --name baseline_0_1__0_6 | tee tmp_output.txt
```

With some of the output being:

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


### Fine-tuning

To fine-tune Stochastic-YOLO, the following commands were used (for dropout rates 25%, 50%, and 75%):
```bash
python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop25 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop50 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop50.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop75 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110
```

Note the use of the `--dropout_ids` flag. Basically, we are using the best YOLOv3 trained model (`best_coco2017_scratch_round2.pt`) but with a non-matching configuration file (`yolov3-mcdropXX.cfg`). Therefore, this flag will indicate where are the dropout ideas in the `.cfg` file (starting from zero, following Darknet format).

[TODO:] point for all scripts inside scripts/ for full documentation

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

python -c "from utils import utils; utils.generate_suppl_latex_table(['mAP', 'PDQ', 'avg_label', 'avg_spatial'])" | tee latex_suppl_output.txt


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