# medword #
Tools to create and evaluate word vector embeddings for medical language.

## Getting started

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

1)  create the same data structure under

        '.../medword/dev_data/'

            /medword/dev_data/embeddings/
            /medword/dev_data/validation_data/

            /medword/dev_data/train_data/
            /medword/dev_data/train_data/raw_data/

    and use smaller raw_data.

2)  change in the configuration.json file the entry "running_mode" to "develop"

        "running_mode": "develop"


#### train data:
get the 'wiki_data' folder  (*.txt files) from Arun, place them into the folder 'raw_data'
(the train_data itself may contain any subdirectories, eg. 'wiki_data', 'wiki_data/unlabeled/'
or 'wiki_data/positives/')

#### validation data:
get validation data from Fabian



## run the following statement

    python medword_pipeline.py


## usefull commands for configuration update

    config.config = json.load(open('configuration.json', 'r'))

    config.config['attribute'] = new_value

    with open('configuration.json', 'w') as f:
        json.dump(config.config, f, indent=4)





