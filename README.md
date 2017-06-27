# medword #
Tools to create and evaluate word vector embeddings for medical language.

## Getting started

### word2vec

#### installation 

    pip install word2vec
    

### fastText (character n-grams by Facebook)

#### installation 

1) chose a basefolder to install fastText (called BASE_DIR in the following)

        cd BASE_DIR
        git clone https://github.com/facebookresearch/fastText.git
        cd fastText
        make

    there should be an executable under "BASE_DIR/fastText/fasttext" now.

2) add

        "fasttext_source": "BASE_DIR/fastText/fasttext"
   
   to the config file.

### configuration

change the filename 'configuration.json.example' to 'configuration.json'

#### adapt the configuration file (configuration.json) if needed

if you want to produce a new train_data file from your data directory

    "compute_new_data": true


if you want to train a new word2vec model from your train_data file

    "train_new_model": true


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



## run the following statement

    python medword_pipeline.py
    
## interactive mode with IPython notebook

    ipython notebook validation_interactive.ipynb


## usefull commands for configuration update

    config.config = json.load(open('configuration.json', 'r'))

    config.config['attribute'] = new_value

    with open('configuration.json', 'w') as f:
        json.dump(config.config, f, indent=4)




   
