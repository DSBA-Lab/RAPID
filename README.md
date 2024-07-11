# RAPID
**RAPID: Training-free Retrieval-based Log Anomaly Detection with PLM considering Token-level information**

[Gunho No](https://www.linkedin.com/in/%EA%B1%B4%ED%98%B8-%EB%85%B8-58b4a9298/)*, [Yukyung Lee](https://www.linkedin.com/in/yukyung-lee-149681155/)\*, [Hyeongwon Kang](https://www.linkedin.com/in/hyeongwon/) and [Pilsung Kang](https://github.com/pilsung-kang) 
<br>(*equal contribution)

This repository is the official implementation of "RAPID".

![RAPID Architecture](img/RAPID_main.png)

```
RAPID/
│
├── split_data.py            # Dataset splitting and preprocessing
├── preprocess_rep.py        # Log representation generation via Language Model
├── ad_test_coreSet.py       # Anomaly detection algorithm
├── utils.py                 
│
├── scripts/
│   └── all_at_once.sh       # End-to-end experiment runner
│
├── processed_data/          # Directory for processed datasets
│   ├── bgl/
│   ├── tbird/
│   └── hdfs/
```

## Datasets
RAPID is evaluated on three public datasets:

* BGL (Blue Gene/L)
* Thunderbird
* HDFS

Place the raw datasets in the dataset/ directory before running the preprocessing scripts.

## Running the Experiments
### Full Pipeline
To reproduce all experiments from the paper:
```
bash scripts/all_at_once.sh
```
This script runs the entire pipeline, including data preprocessing, representation generation, and anomaly detection across multiple configurations as described in our paper.

### Step-by-step Execution

1. Data Splitting and Preprocessing:
```
python split_data.py --dataset [bgl/tbird/hdfs] --test_size 0.2
```

2. Get Representation:
```
python preprocess_rep.py --dataset [bgl/tbird/hdfs] --plm bert-base-uncased --batch_size 8192 --max_token_len [128/512]
```

3. Anomaly Detection:
```
python ad_test_coreSet.py --dataset [bgl/tbird/hdfs] --train_ratio 1 --coreSet 0.01 --only_cls False
```

## Key Parameters

* `--dataset`: Choose the dataset (bgl, tbird, hdfs)
* `--sample`: Sample size for large datasets (e.g., 5000000 for Thunderbird)
* `--plm`: Pre-trained language model (bert-base-uncased, roberta-base, google/electra-base-discriminator)
* `--coreSet`: Core set size or ratio (0, 0.01, 0.1, etc.)
* `--train_ratio`: Ratio of training data to use (1, 0.1, 0.01, etc.)
* `--only_cls`: Whether to use only the CLS token representation (True/False)

## Results
After running the experiments, results will be saved in the `processed_data/[dataset]/[test_size]/[plm]/results/` directory. Each experiment produces a CSV file and a JSON file with detailed performance metrics.

## Citation
If you find this code useful for your research, please cite our paper:

```
@article{NO2024108613,
title = {Training-free retrieval-based log anomaly detection with pre-trained language model considering token-level information},
journal = {Engineering Applications of Artificial Intelligence},
volume = {133},
pages = {108613},
year = {2024},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2024.108613},
url = {https://www.sciencedirect.com/science/article/pii/S0952197624007711},
author = {Gunho No and Yukyung Lee and Hyeongwon Kang and Pilsung Kang}
}
```
