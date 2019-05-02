# Robust Neural Networks are More Interpretable for Genomics

This is a repository that contains datasets and scripts to reproduce the results of "Robust Neural Networks are More Interpretable for Genomics" by Peter K. Koo, Sharon Qian, Gal Kaplun, Verena Volf, and Dimitris Kalimeris. Code is contributed from all authors on this paper. Email: peter_koo@harvard.edu for questions.

The code here depends on Deepomics, a custom-written, high-level APIs written on top of Tensorflow to seamlessly build, train, test, and evaluate neural network models.  WARNING: Deepomics is a required sub-repository.  To properly clone this repository, please use: 

$ git clone --recursive \url{https://github.com/p-koo/uncovering_regulatory_codes.git}

#### Dependencies
* Tensorflow r1.0 or greater (preferably r1.4 or r1.5)
* Python dependencies: PIL, matplotlib, numpy, scipy, sklearn


## Overview of the code

To generate datasets:
* code/0_Generate_Synthetic_dataset.ipynb 
Default is: In [9]: num_seq = 30000
To make smaller dataset, change this to 10000

To train the models on the synthetic dataset: 
* code/1_train_models.py 

These scripts loop through all models described in the manuscript.  Each model can be found in /code/models/

To evaluate the performance of each model on the test set: 
* code/2_print_performance.py 

To quantify interpretability with TPS and FPS scores, run:
* code/3_interpretability_analysis.py
The output is a pickle file that contains a dictionary of the results.

To train, test, and analyze interpretability of models with adversarial training, run the notebook:
* code/4_Adversarial_training.ipynb

## Overview of data

* Due to size restrictions, the dataset is not included in the repository.  Each dataset can be easily created by running the python notebook: 0_Generate_Synthetic_dataset.ipynb 
* pfm_vertebrates.txt contains JASPAR motifs. This is the file that is used as ground truth for the synthetic dataset.

## Overview of results

* All results for each CNN model and dataset are saved in a results directory. 
* Trained model parameters are saved in results/model_params.  
* A reported performance table is saved in results/performance_summary.tsv (automatically outputted from 2_print_performance.py)


