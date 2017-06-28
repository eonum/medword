# medword #
Tools to create and evaluate word vector embeddings for medical language.

## Getting started
Install the required libraries using:

    pip install -r requirements.txt
or

    pip3 install -r requirements.txt


### data

#### create the following data structure:

    '.../medword/data/'

        /medword/data/embeddings/
        /medword/data/validation_data/

        /medword/data/train_data/
        /medword/data/train_data/raw_data/

#### Developper Mode:

if you want to develop on a smaller raw_data set:

1)  create the same data structure (without validation_data folder) under

        '.../medword/dev_data/'

            /medword/dev_data/embeddings/

            /medword/dev_data/train_data/
            /medword/dev_data/train_data/raw_data/

    and use smaller raw_data.

2)  change in the configuration.json file the entry "running_mode" from "normal" to "develop"

        replace "running_mode": "normal"

        with    "running_mode": "develop"


#### train data:
get the 'wiki_data' folder  (*.txt files) from Arun, place them into the folder 'raw_data'
(the train_data itself may contain any subdirectories, eg. 'wiki_data', 'wiki_data/unlabeled/'
or 'wiki_data/positives/')

#### validation data:
get validation data from Fabian


### configuration

1) change the filename 'configuration.json.example' to 'configuration.json'

2) adapt the configuration file (configuration.json) if needed

#### configuration parameters
Running mode: if run in develop mode, the base data directory will be different
(eg. .../dev_data/ instead of .../data/, used to develop on smaller data).

    "running_mode": "normal" or "develop",

Choose embedding method and algorithm. Currently, there are two embedding methods
supported, fasttext

    "embedding_method": "fasttext" or "word2vec",
    "embedding_algorithm": "skipgram" or "cbow",
 
Training file (already preprocessed/tokenized). This is the output file in case 
you process new train data and the input file for the training algorithm.
    
    "train_data_filename": "train_nst.txt",
    
Embedding Model: Output file of the training and input file of the validation.

    "embedding_model_filename": "emb_model_full_ft.bin",

Data directories: 

    "base_data_dir": "data/",
    "develop_base_data_dir": "dev_data/",
    "val_data_dir": "data/validation_data/",
    "config_path": "configuration.json",

Model settings: 

    "embedding_vector_dim": 400,
    "min_token_appearance": 1,
    "tokenizer": "nst" or "sgt",

Validation settings:

    "synonyms_numb_closest_vec": 40,
    "doesntfit_file": "german_doesntfit1.txt",
    "synonyms_file": "german_synonyms3.txt"



## run the following statement

    python medword_pipeline.py
    
## interactive mode with IPython notebook

    ipython notebook validation_interactive.ipynb





   
