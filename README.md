[![license](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/tjiagoM/stochastic-YOLO/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2009.02967-b31b1b.svg)](https://arxiv.org/abs/2009.02967)
# Stochastic-YOLO
*Tiago Azevedo, René de Jong, Partha Maji*

This repository contains all the code necessary to run and further extend the experiments presented in the following ArXiv preprint: [https://arxiv.org/abs/2009.02967](https://arxiv.org/abs/2009.02967)

## Abstract

In image classification tasks, the evaluation of models' robustness to increased dataset shifts with a probabilistic framework is very well studied. However, Object Detection (OD) tasks pose other challenges for uncertainty estimation and evaluation. For example, one needs to evaluate both the quality of the label uncertainty (i.e., *what?*) and spatial uncertainty (i.e., *where?*) for a given bounding box, but that evaluation cannot be performed with more traditional average precision metrics (e.g., mAP). In this paper, we adapt the well-established YOLOv3 architecture to generate uncertainty estimations by introducing stochasticity in the form of Monte Carlo Dropout (MC-Drop), and evaluate it across different levels of dataset shift. We call this novel architecture Stochastic-YOLO, and provide an efficient implementation to effectively reduce the burden of the MC-Drop sampling mechanism at inference time. Finally, we provide some sensitivity analyses, while arguing that Stochastic-YOLO is a sound approach that improves different components of uncertainty estimations, in particular spatial uncertainties.

## Repository preliminaries

This repository was originally forked from https://github.com/ultralytics/yolov3

### Python environment

We include a working dependency file named `environment_yolo_env.yml` describing the exact dependencies used to run this repository. In order to install all the dependencies automatically with [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://anaconda.org/), one can easily just run the following command in the terminal to create an Anaconda environment:

```bash
$ conda env create -f environment_yolo_env.yml
$ conda activate yolo_env
```

### External packages

For this repo it was decided to not install pycocotools in the python environment for a better (and tracked) flexibility when editing some classes. Folder `cocoapi` is then a fork from the original [repository](https://github.com/cocodataset/cocoapi), installed locally in the form of a git submodule (check its README to install cocoapi locally). To import it, it is only necessary to include the folder in the system path whenever needed (e.g. in `test.py`):

```python
import sys
sys.path.append('./cocoapi/PythonAPI/')
```

We also forked an [external repository](https://github.com/david2611/pdq_evaluation) for calculation of the PDQ score, but included as submodule in this repo for flexibility. Instructions there: https://github.com/tjiagoM/pdq_evaluation (originally for the Robotic Vision Challenge 1 and adapted for COCO dataset).

Similarly to `cocoapi`, instead of installing it as a python package, one just needs to include the folder in the system path:

```python
import sys
sys.path.append('./pdq_evaluation')
```

### Repository structure

This repository follows the structure of the original repository, with some changes:

 * `cgf`: Where all the (Darknet) configuration files are. For this repository in specific, `yolov3-custom.cfg` and `yolov3-mcdropXX.cfg` are the ones added.
 * `cocoapi` and `pdq_evaluation`: git submodules with the external packages as defined in previous section.
 * `data`: configuration files for information about the data used in the different scripts. For this repository, we are mostly using `coco2017.data` for the COCO 2017 dataset. We included an extra key named `instances_path` which is not present in the `.data` files from the original Ultalytics' repository.
 * `output`: Mostly temporary output files from different runs.
 * `results`: The CSVs with the metrics calculated from `pdq_evaluation/evaluate.py`. Decision of having one CSV in separate for each model and corruption/severity was due to flexibility in calling parallel evaluation scripts.
 * `scripts`: some bash scripts to run over the terminal for several python tasks.
 * `weights`: Folder with trained models.


## Overall commands

### Training from scratch on COCO 2017

To train YOLOv3 from scratch, with image size input of 416x416 (defined in `yolov3-custom.cfg`), including scaling of images in the train set for 100 epochs:

```bash
python train.py --data data/coco2017.data --epochs 100 --batch-size 20 --name coco2017_scratch --weights '' --cfg cfg/yolov3-custom.cfg --img-size 416
```

After these 100 epochs, and given we passed a `--name` flag, there will be a file in the `output/` folder with the evolution of the training metrics named `results_coco2017_scratch.txt`. 

We further trained the model for another 100 epochs:
```bash
python train.py --data data/coco2017.data --epochs 200 --batch-size 20 --name coco2017_scratch_round2 --weights weights/last_coco2017_scratch.pt --cfg cfg/yolov3-custom.cfg --img-size 416
```

The same process was used to train the five different YOLOv3's for the Ensemble. The commands used for that are inside the `scripts/` folder: `train_ensembles.sh` and `train_ensembles_round2.sh`


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

The `pdq_evaluation` repository was also changed to create a `.csv` file inside `results/` everytime the previous script is executed. Indeed, as you will see for some of the next instructions, the scripts are run having that underlying assumption.

### Fine-tuning

To fine-tune Stochastic-YOLO, the following commands were used (for dropout rates 25%, 50%, and 75%):
```bash
python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop25 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop50 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop50.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110

python train.py --data data/coco2017.data --epochs 50 --batch-size 20 --name coco2017_mcdrop75 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop75.cfg --img-size 416 --dropout_ids 80 82 94 96 108 110
```

Note the use of the `--dropout_ids` flag. Basically, we are using the best YOLOv3 trained model (`best_coco2017_scratch_round2.pt`) but with a non-matching configuration file (`yolov3-mcdropXX.cfg`). Therefore, this flag will indicate where are the dropout ids in the `.cfg` file (starting from zero, following Darknet format).

### Evaluations

The previous evaluations were automated for other models in a few scripts inside the `scripts/` folder, whose contents should be self explanatory. The many commands are scattered across different files instead of a single one to make it easy to parallelise these commands across different servers. In any case, if something is not clear please open a new issue in this repository. Some scripts are repeated with just different parameters.

One starting point can be `generate_corruptions_for_higher_drops.sh` where evaluation for Stochastic-YOLO with dropout rates of 50% and 75% are performed across different corruptions and severities, as well as for fine-tuned and non fine-tuned models. One can see the usage of some flags: `--dropout-at_inference` to activate dropout at inference time, `--num_samples` to indicate how many times the model will be sampled from, `--corruption_num` to indicate which corruption number will be used (according to [imagecorruptions package](https://github.com/bethgelab/imagecorruptions)), and `--severity` which is a severity number also described in the [imagecorruptions package](https://github.com/bethgelab/imagecorruptions) (between 1 and 5). `confidence_0_1.sh` and `confidence_0_1_higherdrops.sh` are similar scripts for 0.1 confidence threshold.

For the sensitivity analysis on dropout rate, the script used was `drop_rates_eval.sh`. Note that for this case, instead of creating extra `.cfg` files, it was used a new flag `--new_drop_rate` so the model's dropout rate would be changed at runtime after loading some model with a different dropout rate.

Although not present in these scripts, to activate the caching mechanism described in the paper, one just has to pass the flag `--with_cached_mcdrop` to `test.py`.

Evaluations for the Ensembles can be achieved with two scripts (one for each confidence threshold):

```bash
$ bash scripts/ensembles_0_5.sh
$ bash scripts/ensembles_0_1.sh
```

### Plotting results

The following command will plot many metrics across dropout rates and models, with one plot for each corruption (not present in the paper):
```bash
$ bash scripts/plot_corruption_results.sh
```

For the plots used in the paper, this is the command:
```bash
python -c "from utils import utils; utils.generate_paper_plots()"
```

### LaTeX tables for paper

python -c "from utils import utils; utils.generate_latex_table(['mAP', 'PDQ', 'avg_label', 'avg_spatial'])" | tee latex_output.txt

python -c "from utils import utils; utils.generate_suppl_latex_table(['mAP', 'PDQ', 'avg_label', 'avg_spatial'])" | tee latex_suppl_output.txt



### Getting Unknowns

One set of experiments which is not present in the paper, is the analysis of unknowns. If you pass the flag `--get_unknowns` to `test.py` like in the following two examples: 

```bash
python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name unknown_tests --get_unknowns --batch-size 40

python test.py --data data/coco2017.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-mcdrop25.cfg --conf-thres=0.5 --iou-thres=0.6  --dropout_ids 80 82 94 96 108 110 --dropout_at_inference --num_samples 10 --name unknown_testsMCD25NO --get_unknowns --batch-size 20
```

The returning bounding boxes will be instead what we consider to be "unknowns" or, in other words, those bounding boxes in which the model is not entirely sure what object is inside. In practice, this corresponds to filter out bounding boxes where the label scores are all below 0.5 (and above 0.1 just from a practical point of view). In specific, this filtering is done inside `utils.non_max_suppression()`.

We also preliminarly explored an external dataset [TACO](https://github.com/pedropro/TACO) to see how the models would capture unknowns in an OOD dataset. As we didn't want to calculate any metrics for this dataset (only qualitatively see what bounding boxes were being predicted) we created a new flag `--only_inference` that can be passed to `test.py`, to indicate that no metrics need to be calculated. This is necessary to maintain compatibility. These are two examples of how to execute this flag:

```bash
python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.5 --iou-thres=0.6 --name taco_experiment --batch-size 40 --only_inference

python test.py --data data/taco_batch13.data --img-size 416 --weights weights/best_coco2017_scratch_round2.pt --cfg cfg/yolov3-custom.cfg --conf-thres=0.8 --iou-thres=0.6 --name taco_experiment --batch-size 40  --get_unknowns --only_inference
```

Notice that we provide some files inside `data/` for a reduced amount of images from TACO's repository in a format needed by this repository.