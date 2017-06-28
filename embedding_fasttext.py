"""
embedding method: FastText by Facebook

- using subword information: character n-grams

https://github.com/salestock/fastText.py
https://pypi.python.org/pypi/fasttext

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
        """
        Train fasttext model with following parameters:
        ***********************************************

        input_file     training file path (required)
        output         output file path (required)
        lr             learning rate [0.05]
        lr_update_rate change the rate of updates for the learning rate [100]
        dim            size of word vectors [100]
        ws             size of the context window [5]
        epoch          number of epochs [5]
        min_count      minimal number of word occurences [5]
        neg            number of negatives sampled [5]
        word_ngrams    max length of word ngram [1]
        loss           loss function {ns, hs, softmax} [ns]
        bucket         number of buckets [2000000]
        minn           min length of char ngram [3]
        maxn           max length of char ngram [6]
        thread         number of threads [12]
        t              sampling threshold [0.0001]
        silent         disable the log output from the C++ extension [1]
        encoding       specify input_file encoding [utf-8]
        """

        print("\nEmbedding Algorithm: fastText pure")

        # remove extention for fasttext
        model_name, ext = os.path.splitext(emb_model_fn)
        emb_model_src_no_ext = os.path.join(emb_model_dir, model_name)

        # Model parameters
        algorithm = "skipgram"      # skipgram or cbow


        # train model
        if algorithm is "skipgram":
            self._model = fasttext.skipgram(train_data_src, emb_model_src_no_ext, silent=0)
            self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec')
        elif algorithm is "cbow":
            self._model = fasttext.cbow(train_data_src, emb_model_src_no_ext)
            self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec', silent=0)
        else:
            print("fasttext algorithm must be 'skipgram' or 'cbow' ")
            return AttributeError


        # save configuration
        config_fn = model_name + '_configuration.json'
        config_src = os.path.join(emb_model_dir, config_fn)

        with open(config_src, 'w') as f:
            json.dump(self.config.config, f, indent=4)

        # get total number of words in trainfile
        n_lines = 0
        n_words = 0
        with open(train_data_src) as f:
            for line in f.readlines():
                n_lines += 1
                n_words += len(line.split())

        # note that word2vec counts newline chars as words, we do the same here for consistency
        print("Training finsihed.\n"
              "Vocab size:", len(self.get_vocab()), ", words in train file:", n_lines + n_words, "\n"
              "Model saved at:", emb_model_src_no_ext + ".bin,", emb_model_src_no_ext + ".vec" )



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

