import os
import importlib

import preprocess as pp
import embedding
import model_validation as mv

# reload imported modules, as older versions are cached (developing purpose)
importlib.reload(pp)
importlib.reload(embedding)
importlib.reload(mv)

### script settings ###
# if you want to produce a new train_data file from your data directory
COMPUTE_NEW_TRAIN_DATA = True

# if you want to train a new word2vec model from your train_data file
TRAIN_NEW_MODEL = False

# directory
DATA_DIR = 'data/embeddings/'
pp.make_directory('data', 'embeddings')


# filenames
train_data_fn = 'train_trial.txt'
emb_model_fn = 'emb_model.bin'

# paths for train_data and output files
train_fp = os.path.join(DATA_DIR, train_data_fn)
emb_model_fp = os.path.join(DATA_DIR, emb_model_fn)


# compute new train data if needed
if (COMPUTE_NEW_TRAIN_DATA):
    pp.create_train_data(train_fp)


# train embeddings using word2vec
if (TRAIN_NEW_MODEL):
    embedding.make_emb_from_file(train_fp, emb_model_fp)


# validate the embedding model
model = mv.validate_model(emb_model_fp)



