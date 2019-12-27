# Robust Neural Networks are More Interpretable for Genomics

This is a repository that contains datasets and scripts to reproduce the results of "Robust Neural Networks are More Interpretable for Genomics" by Peter K. Koo, Sharon Qian, Gal Kaplun, Verena Volf, and Dimitris Kalimeris, which was presented at the ICML Workshop for Computational Biology 2019 in Long Beach, CA. A preprint for this work can be found via https://www.biorxiv.org/content/10.1101/657437v1. 

Code is contributed from all authors on this paper. Email: peter_koo@harvard.edu for questions. The code here depends on Deepomics, a custom-written, high-level APIs written on top of Tensorflow to seamlessly build, train, test, and evaluate neural network models.  WARNING: Deepomics is a required sub-repository.  To properly clone this repository, please use: 

$ git clone --recursive \url{https://github.com/p-koo/uncovering_regulatory_codes.git}

#### Dependencies
* Tensorflow r1.0 or greater (preferably r1.14 or r1.15)
* Python dependencies: PIL, matplotlib, numpy, scipy (version 1.1.0), sklearn
* meme suite (5.1.0)

## Overview of the code

To generate datasets:
* code/0_Generate_Synthetic_dataset.ipynb 
Default is: In [9]: num_seq = 30000
To make smaller dataset, change this to 10000

To train all models on the synthetic dataset (without/with regularization and with gaussian noise): 
* code/1_train_all_models.py 

To train all models on the synthetic dataset (with adversarial training):
* code/1_train_all_models_adv.py 

These scripts loop through all models described in the manuscript.  Each model can be found in /code/models/

To train a single model on synthetic dataset, use:
* code/train_single_model.py 
* code/train_single_model_adv.py 

To evaluate the performance of each model on the test set: 
* code/2_print_performance.py 

To quantify interpretability:
* code/3_interpretability_analysis_all_models.py
* code/3_interpretability_analysis_single_model.py
The output is a pickle file that contains a dictionary of the results.



## Overview of data

* Due to size restrictions, the dataset is not included in the repository.  Each dataset can be easily created by running the python notebook: 0_Generate_Synthetic_dataset.ipynb 
* pfm_vertebrates.txt contains JASPAR motifs. This is the file that is used as ground truth for the synthetic dataset.

## Overview of results

* All results for each CNN model and dataset are saved in a results directory. 
* Trained model parameters are saved in results/model_params. 
* A reported performance table is saved in results/performance_summary.tsv (automatically outputted from 2_print_performance.py)


