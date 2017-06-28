# from shared.load_config import load_config
from shared.load_config import __CONFIG__


import os
import importlib
import json


import preprocess as pp
import embedding_fasttext
import embedding_fasttext_gensim
import embedding_word2vec
import model_validation as mv

# reload imported modules, as older versions are cached (developing purpose)
importlib.reload(pp)
importlib.reload(mv)
importlib.reload(embedding_fasttext)
importlib.reload(embedding_word2vec)

def run_pipeline(config):

    # setup needed libraries, data structures etc.
    pp.setup()

    # choose the embedding algorithm
    implementation = config['embedding_algorithm']
    if implementation == 'fasttext':
        embedding = embedding_fasttext.EmbeddingFasttext(config)

    elif implementation == 'word2vec':
        embedding = embedding_word2vec.EmbeddingWord2vec(config)

    else:
        print('embedding_algorithm (in config) must be "fasttext" or "word2vec"')
        return AttributeError


    ### script settings ###
    # if you want to produce a new train_data file from your data directory
    COMPUTE_NEW_TRAIN_DATA = config.config['compute_new_data']

    # if you want to train a new word2vec _model from your train_data file
    TRAIN_NEW_MODEL = config.config['train_new_model']

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
    emb_model_src = os.path.join(emb_model_dir, emb_model_fn)

    # if not exists make embeddings folder
    if not os.path.exists(emb_model_dir):
        os.makedirs(emb_model_dir)

    # source paths for train_data
    train_data_dir = os.path.join(base_data_dir, 'train_data/')
    train_data_fn = config.config['train_data_filename']
    train_data_src = os.path.join(train_data_dir, train_data_fn)


    # compute new train data if needed
    if (COMPUTE_NEW_TRAIN_DATA):
        raw_data_dir = os.path.join(train_data_dir, 'raw_data/')
        pp.create_train_data(train_data_src, raw_data_dir, config)


    # train embeddings
    if (TRAIN_NEW_MODEL):
        embedding.train_model(train_data_src, emb_model_dir, emb_model_fn)

    # validate the embedding _model
    mv.validate_model(embedding, emb_model_dir, emb_model_fn)


    return embedding

if __name__ == '__main__':

    model = run_pipeline(__CONFIG__) #TODO remove return value, only for development
    print("end_main")


