"""
each new embedding method (like word2vec or fastText) must inherit from
the class EmbeddingBaseAbstract and implement all its functions

"""

from abc import ABC, abstractmethod

class EmbeddingAbstractBase(ABC):

    @abstractmethod
    def __init__(self, config):
        # initialize object variables like self._model
        pass

    @abstractmethod
    def train_model(self, train_data_src, emb_model_dir, emb_model_fn):
        # train a model and store it in embeddig object
        pass


    @abstractmethod
    def similarity(self, word1, word2):
        # return cosine similarity of word1 and word2
        pass

    @abstractmethod
    def most_similar_n(self, word, topn):
        # return a list of the n most similar words in model
        pass

    @abstractmethod
    def get_vocab(self):
        # return a list of vocabulary
        pass

    @abstractmethod
    def word_vec(self, word):
        # return an array of the embedding vector belonging to 'word'
        pass

    @abstractmethod
    def may_construct_word_vec(self, word):
        # return True if model can get e vector for 'word'
        pass

    @abstractmethod
    def load_model(self, emb_model_dir, emb_model_fn):
        # load embedding model
        pass

    @abstractmethod
    def analogy(self, positives, negatives, topn):
        # load embedding model
        pass

    @abstractmethod
    def vec_dim(self):
        # return dimension of embedding vectors
        pass



