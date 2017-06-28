"""
embedding model: FastText by Facebook

- using subword information: character n-grams

https://pypi.python.org/pypi/fasttext#enriching-word-vectors-with-subword-information

"""
import fasttext
import numpy as np
import os
import json
from embedding_base import EmbeddingBaseAbstract
from gensim.models import KeyedVectors

class EmbeddingFasttext(EmbeddingBaseAbstract):

    def __init__(self, config):
        self._model = None
        self._vectors = None
        self.config = config


    def train_model(self, train_data_src, emb_model_dir, emb_model_fn):
        print("\nEmbedding Algorithm: fastText pure")

        # remove extention for fasttext
        model_name, ext = os.path.splitext(emb_model_fn)
        emb_model_src_no_ext = os.path.join(emb_model_dir, model_name)

        # Model parameters
        algorithm = "skipgram"      # skipgram or cbow


        # train model
        if algorithm is "skipgram":
            self._model = fasttext.skipgram(train_data_src, emb_model_src_no_ext)
            self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec')
        elif algorithm is "cbow":
            self._model = fasttext.cbow(train_data_src, emb_model_src_no_ext)
            self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec')
        else:
            print("fasttext algorithm must be 'skipgram' or 'cbow' ")
            return AttributeError


        # save configuration
        config_fn = model_name + '_configuration.json'
        config_src = os.path.join(emb_model_dir, config_fn)

        with open(config_src, 'w') as f:
            json.dump(self.config.config, f, indent=4)

        print("Training finsihed. \nModel saved at:", emb_model_src_no_ext + ".bin,", emb_model_src_no_ext + ".vec" )



    def similarity(self, word1, word2):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.cosine_similarity(word1, word2)


    def most_similar_n(self, word, n=10):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        word_vec = self.word_vec(word)

        return self._vectors.similar_by_vector(word_vec)


    def load_model(self, emb_model_dir, emb_model_fn):
        model_name, ext = os.path.splitext(emb_model_fn)
        emb_model_src_no_ext = os.path.join(emb_model_dir, model_name)

        self._model = fasttext.load_model(emb_model_src_no_ext + '.bin')
        self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec')

    def get_vocab(self):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return list(self._model.words)


    def word_vec(self, word):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return np.asarray(self._model[word], dtype=np.float64)


    def may_construct_word_vec(self, word):

        return True

