"""
embedding method: FastText by Facebook

- using subword information: character n-grams

wrapper: https://github.com/salestock/fastText.py (pip install fasttext)
https://pypi.python.org/pypi/fasttext

"""
import fasttext
import numpy as np
import os
import json
from embedding_base import EmbeddingAbstractBase
from gensim.models import KeyedVectors
import multiprocessing


class EmbeddingFasttext(EmbeddingAbstractBase):

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


        # remove extention for fasttext
        model_name, ext = os.path.splitext(emb_model_fn)
        emb_model_src_no_ext = os.path.join(emb_model_dir, model_name)


        # Model parameters
        algorithm = self.config.config['embedding_algorithm']     # skipgram or cbow
        print("Embedding Method: fastText, Algorithm:", algorithm)

        # embedding vector dimension
        emb_dim = self.config.config['embedding_vector_dim']

        # minimum number a token has to appear to be included in _model
        min_count = self.config.config['min_token_appearance']

        # min ngram length (number of chars)
        minn = 3

        # max ngram length (number of chars)
        maxn = 6

        # number of cores
        n_cores = multiprocessing.cpu_count()

        # train model
        if algorithm == "skipgram":
            self._model = fasttext.skipgram(input_file = train_data_src,
                                            output = emb_model_src_no_ext,
                                            dim = emb_dim,
                                            min_count = min_count,
                                            minn = minn,
                                            maxn = maxn,
                                            thread = n_cores,
                                            silent = 0)

            self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec')

        elif algorithm == "cbow":
            self._model = fasttext.cbow(input_file = train_data_src,
                                        output = emb_model_src_no_ext,
                                        dim = emb_dim,
                                        min_count = min_count,
                                        minn = minn,
                                        maxn = maxn,
                                        thread = n_cores,
                                        silent = 0)

            self._vectors = KeyedVectors.load_word2vec_format(emb_model_src_no_ext + '.vec')

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
        # to compare these two word counts

        print("Training finsihed.\n"
              "Vocab size:", len(self.get_vocab()), ", words in train file:", n_lines + n_words, "\n"
              "Model saved at:", emb_model_src_no_ext + ".bin,", emb_model_src_no_ext + ".vec" )


    def similarity(self, word1, word2):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.cosine_similarity(word1, word2)


    def most_similar_n(self, word, topn=10):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        word_vec = self.word_vec(word)

        return self._vectors.similar_by_vector(word_vec, topn)


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


    def vec_dim(self):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.dim


    def analogy(self, positives, negatives, topn):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._vectors.most_similar(positives, negatives, topn)


    def may_construct_word_vec(self, word):
        # TODO implement, assumes that fasttext can construct all words at the moment
        return True

