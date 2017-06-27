import multiprocessing
import numpy as np
import os
import json
from subprocess import PIPE, Popen
import word2vec as w2v
from embedding_base import EmbeddingBaseAbstract


class EmbeddingWord2vec(EmbeddingBaseAbstract):

    def __init__(self, config):
        self._model = None
        self.config = config

    def word_vec(self, word):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.get_vector(word)

    def get_vocab(self):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.vocab.tolist()

    def most_similar_n(self, word, n=10):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        indexes, metrics = self._model.cosine(word, n)
        return self._model.generate_response(indexes, metrics).tolist()


    def similarity(self, word1, word2):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        v1 = self._model[word1] / np.linalg.norm(self._model[word1], 2)
        v2 = self._model[word2] / np.linalg.norm(self._model[word2], 2)

        return np.dot(v1, v2)


    def train_model(self, train_data_src, emb_model_dir, emb_model_fn):

        print("\nEmbedding Algorithm: word2vec")

        ### embedding parameters

        # embedding vector dimension
        emb_dim = self.config.config['embedding_vector_dim']

        # minimum number a token has to appear to be included in _model
        min_count = self.config.config['min_token_appearance']

        # number of cores
        n_cores = multiprocessing.cpu_count()

        # embedding _model source
        emb_model_src = os.path.join(emb_model_dir, emb_model_fn)

        # downsampling high occurence
        sample_freq = 1e-5

        print("Start training the model.")

        command = ["word2vec", "-train", train_data_src, "-output", emb_model_src,
                   "-binary", "1", "-cbow", '1', "-size", str(emb_dim), "-sample", str(sample_freq),
                   "-min-count", str(min_count), "-threads", str(n_cores)]

        # Open pipe to subprocess
        proc = Popen(command, stdout=PIPE, stderr=PIPE)

        result_list = []

        # get output from subprocess
        while proc.poll() is None:

            while proc.poll() is None:
                i = proc.stdout.read(1).decode('ascii')
                result_list.append(i)
                if i == "\n" or i == "\r":
                    break

            if result_list != []:
                print("".join(result_list), end="")
                result_list = []

        # save config file
        filename, ext = os.path.splitext(emb_model_fn)
        config_fn = filename + '_configuration.json'
        config_src = os.path.join(emb_model_dir, config_fn)

        with open(config_src, 'w') as f:
            json.dump(self.config.config, f, indent=4)

        # load newly generated model
        self.load_model(emb_model_dir, emb_model_fn)

        print("Training finsihed. \nModel saved at:", emb_model_src)

        """
        word2vec execution
        Parameters for training:
            train <file_path>
                Use text data from <file_path> to train the _model
            output <file_path>
                Use <file_path> to save the resulting word vectors / word clusters
            size <int>
                Set size of word vectors; default is 100
            window <int>
                Set max skip length between words; default is 5
            sample <float>
                Set threshold for occurrence of words. Those that appear with
                higher frequency in the training data will be randomly
                down-sampled; default is 0 (off), useful value is 1e-5
            hs <int>
                Use Hierarchical Softmax; default is 1 (0 = not used)
            negative <int>
                Number of negative examples; default is 0, common values are 5 - 10
                (0 = not used)
            threads <int>
                Use <int> threads (default 1)
            iter_ = number of iterations (epochs) over the corpus. Default is 5.
            min_count <int>
                This will discard words that appear less than <int> times; default
                is 5
            alpha <float>
                Set the starting learning rate; default is 0.025
            debug <int>
                Set the debug mode (default = 2 = more info during training)
            binary <int>
                Save the resulting vectors in binary moded; default is 0 (off)
            cbow <int>
                Use the continuous bag of words _model; default is 1 (skip-gram
                _model)
            save_vocab <file>
                The vocabulary will be saved to <file>
            read_vocab <file>
                The vocabulary will be read from <file>, not constructed from the
                training data
            verbose
                Print output from training
        """


    def load_model(self, emb_model_dir, emb_model_fn):
        # load _model
        emb_model_src = os.path.join(emb_model_dir, emb_model_fn)
        self._model = w2v.load(emb_model_src)


def make_emb_from_file(train_data_src, emb_model_dir, emb_model_fn, config):

    print("Not used anymore")

    ### embedding parameters

