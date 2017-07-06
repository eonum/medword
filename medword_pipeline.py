from shared.load_config import __CONFIG__

import os
import importlib
import json


import preprocess as pp
import embedding_fasttext
import embedding_word2vec
import embedding_word2vec_composite
import model_validation as mv

# reload imported modules, as older versions are cached (developing purpose)
importlib.reload(pp)
importlib.reload(mv)
importlib.reload(embedding_fasttext)
importlib.reload(embedding_word2vec)

def run_pipeline(embedding):

    # setup needed libraries, data structures etc.
    pp.setup()

    # get config
    config = embedding.config

    ### script settings ###
    # if you want to produce a new train_data file from your data directory
    COMPUTE_NEW_TRAIN_DATA = config.config['compute_new_data']

    # if you want to train a new word2vec model from your train_data file
    TRAIN_NEW_MODEL = config.config['train_new_model']

    # if you want to run the validation
    RUN_VALIDATION = config.config['run_validation']

    # data directories
    if(config.config['running_mode'] == 'develop'):
        print('Running in DEVELOPPER mode.')
        base_data_dir = config.config['develop_base_data_dir']
    elif(config.config['running_mode'] == 'normal'):
        print('Running in NORMAL mode.')
        base_data_dir = config.config['base_data_dir']
    else:
        print("Running mode not recognized: set running_mode to 'normal' or 'develop'")
        return None

    # source paths for embeddings
    emb_model_dir = os.path.join(base_data_dir, 'embeddings/')
    emb_model_fn = config.config['embedding_model_filename']

    # if not exists make embeddings folder
    if not os.path.exists(emb_model_dir):
        os.makedirs(emb_model_dir)

    # source paths for train_data
    train_data_dir = os.path.join(base_data_dir, 'train_data/')
    train_data_fn = config.config['train_data_filename']
    train_data_src = os.path.join(train_data_dir, train_data_fn)


    # compute new train data if needed
    if (COMPUTE_NEW_TRAIN_DATA):
        print("\n*** COMPUTING TRAIN DATA *** ")
        raw_data_dir = os.path.join(train_data_dir, 'raw_data/')
        pp.create_train_data(train_data_src, raw_data_dir, config)
        print("*** END COMPUTING TRAIN DATA *** ")


    # train embeddings
    if (TRAIN_NEW_MODEL):
        print("\n*** TRAINING NEW MODEL *** ")
        embedding.train_model(train_data_src, emb_model_dir, emb_model_fn)
        print("*** END TRAINING NEW MODEL *** ")

    # validate the embedding model
    if (RUN_VALIDATION):
        print("\n*** VALIDATING MODEL *** ")
        mv.validate_model(embedding, emb_model_dir, emb_model_fn)
        print("*** END VALIDATING MODEL *** ")


if __name__ == '__main__':

    config = __CONFIG__

    # choose the embedding algorithm
    emb_method = config.config['embedding_method']
    if emb_method == 'fasttext':
        embedding = embedding_fasttext.EmbeddingFasttext(config)

    elif emb_method == 'word2vec':
        embedding = embedding_word2vec.EmbeddingWord2vec(config)

    elif emb_method == 'word2vec-compound':
        embedding = embedding_word2vec_composite.EmbeddingWord2vecComposite(config)

    else:
        print('embedding_algorithm (in config) must be "fasttext" or "word2vec"')
        raise AttributeError

    run_pipeline(embedding)

    print("end_main")


