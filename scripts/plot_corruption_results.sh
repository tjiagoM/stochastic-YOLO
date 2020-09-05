#!/bin/bash

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '50%', '75%'], metrics= ['mAP'], save_path='plots/baseline_vs_mcdrop_0_5_map.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '50%', '75%'], metrics= ['PDQ'], save_path='plots/baseline_vs_mcdrop_0_5_pdq.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '50%', '75%'], metrics= ['avg_spatial', 'avg_label'], save_path='plots/baseline_vs_mcdrop_0_5_spatial_label.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '25% no retrain', '50%', '50% no retrain', '75%', '75% no retrain'], metrics= ['avg_label'], save_path='plots/mcdrops_0_5_label.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '25% no retrain', '50%', '50% no retrain', '75%', '75% no retrain'], metrics= ['avg_spatial'], save_path='plots/mcdrops_0_5_spatial.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '25% no retrain', '50%', '50% no retrain', '75%', '75% no retrain'], metrics= ['mAP'], save_path='plots/mcdrops_0_5_map.png', params_name = '0_5_0_6')"



python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '50%', '75%'], metrics= ['mAP'], save_path='plots/baseline_vs_mcdrop_0_1_map.png', params_name = '0_1_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '50%', '75%'], metrics= ['PDQ'], save_path='plots/baseline_vs_mcdrop_0_1_pdq.png', params_name = '0_1_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '50%', '75%'], metrics= ['avg_spatial', 'avg_label'], save_path='plots/baseline_vs_mcdrop_0_1_spatial_label.png', params_name = '0_1_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '25% no retrain', '50%', '50% no retrain', '75%', '75% no retrain'], metrics= ['avg_label'], save_path='plots/mcdrops_0_1_label.png', params_name = '0_1_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '25% no retrain', '50%', '50% no retrain', '75%', '75% no retrain'], metrics= ['avg_spatial'], save_path='plots/mcdrops_0_1_spatial.png', params_name = '0_1_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop25_10', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '25%', '25% no retrain', '50%', '50% no retrain', '75%', '75% no retrain'], metrics= ['mAP'], save_path='plots/mcdrops_0_1_map.png', params_name = '0_1_0_6')"



python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop01_10_no_retrain', 'coco_mcdrop05_10_no_retrain', 'coco_mcdrop10_10_no_retrain', 'coco_mcdrop15_10_no_retrain', 'coco_mcdrop20_10_no_retrain', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '1%', '5%', '10%', '15%', '20%', '25%', '50%', '75%'], metrics= ['avg_spatial'], save_path='plots/mcdroprates_0_5_spatial.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop01_10_no_retrain', 'coco_mcdrop05_10_no_retrain', 'coco_mcdrop10_10_no_retrain', 'coco_mcdrop15_10_no_retrain', 'coco_mcdrop20_10_no_retrain', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '1%', '5%', '10%', '15%', '20%', '25%', '50%', '75%'], metrics= ['avg_label'], save_path='plots/mcdroprates_0_5_label.png', params_name = '0_5_0_6')"

python -c "from utils import utils; utils.plot_corrupted_results(model_names=['coco_baseline', 'coco_mcdrop01_10_no_retrain', 'coco_mcdrop05_10_no_retrain', 'coco_mcdrop10_10_no_retrain', 'coco_mcdrop15_10_no_retrain', 'coco_mcdrop20_10_no_retrain', 'coco_mcdrop25_10_no_retrain', 'coco_mcdrop50_10_no_retrain', 'coco_mcdrop75_10_no_retrain'], label_matches=['baseline', '1%', '5%', '10%', '15%', '20%', '25%', '50%', '75%'], metrics= ['mAP'], save_path='plots/mcdroprates_0_5_map.png', params_name = '0_5_0_6')"