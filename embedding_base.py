from abc import ABC, abstractmethod

class EmbeddingBaseAbstract(ABC):

    @abstractmethod
    def __init__(self, config):
        pass

    @abstractmethod
    def train_model(self, train_data_src, emb_model_dir, emb_model_fn):
        pass

    @abstractmethod
    def similarity(self, word1, word2):
        pass

    @abstractmethod
    def most_similar_n(self, word, n):
        pass

    @abstractmethod
    def get_vocab(self):
        pass

    @abstractmethod
    def word_vec(self, word):
        pass

    @abstractmethod
    def may_construct_word_vec(self, word):
        pass

    @abstractmethod
    def load_model(self, emb_model_dir, emb_model_fn):
        pass


