#!/bin/bash

# THIS SHOULD BE THE PATH TO THE CLONED HUGGING FACE DATASETS GIT REPOSITORY
DATASET_PATH=$1

cd ..
python hf_datasets/generate_code_from_repository.py .
python hf_datasets/split.py ${DATASET_PATH}
rm -fr "/home/lagunas/devel/hf/datasets/datasets/code_x_glue_tc_nl_code_search_web_query"
python hf_datasets/generate_dataset_card.py ${DATASET_PATH}
python hf_datasets/test.py ${DATASET_PATH}
cd ${DATASET_PATH}
make style
flake8 datasets
