#### medword ####
Tools to create and evaluate word vector embeddings for medical language.

### Getting started

## data
# create the following data structure.
    '.../medword/data/'
        /medword/data/embeddings/
        /medword/data/train_data/

# get the 'wiki_data' folder  (*.txt files) from Arun, place them into the folder train_data
# (the train_data itself may contain any subdirectories, eg. 'wiki_data', 'wiki_data/unlabeled/'
# or 'wiki_data/positives/')


## script settings (medword_pipeline.py)
# if you want to produce a new train_data file from your data directory
COMPUTE_NEW_TRAIN_DATA = True

# if you want to train a new word2vec model from your train_data file
TRAIN_NEW_MODEL = True


## run the following statement
python medword_pipeline.py

