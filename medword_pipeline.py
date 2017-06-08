import os
import importlib

import preprocess as pp
import embedding
import model_validation as mv

# reload imported modules, as older versions are cached (developing purpose)
importlib.reload(pp)
importlib.reload(embedding)
importlib.reload(mv)

# script settings
compute_new_train_data = True

# directory
DATA_DIR = '/Users/Fabian/Developer/eonum/Coding/medword/data/embeddings/'

# filenames
train_data_fn = 'train.txt'
emb_model_fn = 'emb_model.bin'

# paths for train_data and output files
train_fp = os.path.join(DATA_DIR, train_data_fn)
emb_model_fp = os.path.join(DATA_DIR, emb_model_fn)


# compute new train data if needed
if (compute_new_train_data):
    pp.create_train_data(train_fp)


# train embeddings using word2vec
embedding.make_emb_from_file(train_fp, emb_model_fp)

# validate the embedding model
model = mv.validate_model(emb_model_fp)



