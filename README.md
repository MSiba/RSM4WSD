# Research on Word Sense Disambiguation using rotating spheres model.

## What?
This repository provides the code to run Word Sense Disambiguation using Rotating Spheres Model.
- **resources**: in this file the training corpora SemCor and PWNGC as well as all evaluation datasets are preprocessed. (Section 5.2.1 in the manuscript)
- **statistics**: this file stores some statistics on the datasets
- **embedding_space**: in this file, you'll find the embedding space construction using MPTT and geometric training (Section 4.2 in the manuscript)
- **data**: this file contains all processed data. It is a very large file, you will find tracker files that indicate in which script each dataset is generated and where it is re-used. Please download all data files from https://drive.google.com/drive/folders/1mPJ127-CfexqijTGRNQj1RR4Z7Fp7pTE?usp=sharing
- **encoder**: the encoder has been trained and tested on Google Colab. To run the code, please refer to section 'Encoder' of this ReadMe file

## How to run the code?
1. download the requirements.txt file
2. If you want to start from the beginning (embedding WordNet into the embedding space), do the following:
  i. transform WordNet into graph, run <wn2graph.py> to parse each POS group.
  ii. 
4. If you want to use previously created data, you can download the processed 'data' file from https://drive.google.com/drive/folders/1mPJ127-CfexqijTGRNQj1RR4Z7Fp7pTE?usp=sharing



# Encoder: Training, validation, testing
## Location: Google ColabPro, and Google ColabProPlus
To reproduce the training procedure:
1. download the 'ColabNotebooks' file to your Google Drive 
2. Link: https://drive.google.com/drive/folders/1YpEY4RsNJEdwYSGfNM7Gib8A9RWqHF-t?usp=sharing
3. open the .ipynb notebook
4. run the cells
5. Check the ouput in the results, and annotations files
