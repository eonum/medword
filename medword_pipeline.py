# from shared.load_config import load_config
from shared.load_config import __CONFIG__


import os
import importlib
import json

import preprocess as pp
import embedding
import model_validation as mv

# reload imported modules, as older versions are cached (developing purpose)
importlib.reload(pp)
importlib.reload(embedding)
importlib.reload(mv)


def run_pipeline(config):

    pp.setup()

    # # edit script settings if needed
    # config.config['running_mode'] = 'normal'
    # config.config['running_mode'] = 'develop'
    #
    # write it back to the file
    # with open('configuration.json', 'w') as f:
    #     json.dump(config.config, f, indent=4)

    ### script settings ###
    # if you want to produce a new train_data file from your data directory
    COMPUTE_NEW_TRAIN_DATA = config.config['compute_new_data']

    # if you want to train a new word2vec model from your train_data file
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

    emb_model_dir = os.path.join(base_data_dir, 'embeddings/')
    train_data_dir = os.path.join(base_data_dir, 'train_data/')

    # if not exists make embeddings folder
    if not os.path.exists(emb_model_dir):
        os.makedirs(emb_model_dir)

    # filenames
    train_data_fn = config.config['train_data_filename']
    emb_model_fn = config.config['embedding_model_filename']

    # source paths for train_data and output files
    train_data_src = os.path.join(train_data_dir, train_data_fn)
    emb_model_src = os.path.join(emb_model_dir, emb_model_fn)


    # compute new train data if needed
    if (COMPUTE_NEW_TRAIN_DATA):
        raw_data_dir = os.path.join(train_data_dir, 'raw_data/wiki_data/positive/')
        pp.create_train_data(train_data_src, raw_data_dir, config)


    # train embeddings using word2vec
    if (TRAIN_NEW_MODEL):
        embedding.make_emb_from_file(train_data_src, emb_model_dir, emb_model_fn, config)

    # validate the embedding model
    model = mv.validate_model(emb_model_src, config)

    return model

if __name__ == '__main__':

    model = run_pipeline(__CONFIG__) #TODO remove return value, only for development
    print("end_main")


