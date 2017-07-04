import numpy as np
import word2vec as w2v
from embedding_word2vec import EmbeddingWord2vec

class EmbeddingWord2vecComposite(EmbeddingWord2vec):

    def __init__(self, config):
        self._model = None
        self.config = config

    def word_vec(self, word):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        if word in self._model.vocab_hash:
            return self._model.get_vector(word)

        end = len(word) - 1
        vector = None

        while word != '' and end > 0:
            if word[:end] in self._model.vocab_hash:
                print(word[:end])
                if vector == None:
                    vector = self._model.get_vector(word[:end])
                else:
                    vector = vector + self._model.get_vector(word[:end])
                word = word[end:]
                end = len(word)
            else:
                end -= 1


        return vector

    def most_similar_n(self, word, topn=10):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        indexes, metrics = self._model.cosine(word, topn)
        return self._model.generate_response(indexes, metrics).tolist()


    def analogy(self, positives, negatives, topn):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        indexes, metrics = self._model.analogy(positives, negatives, topn)
        return self._model.generate_response(indexes, metrics).tolist()


    def similarity(self, word1, word2):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        v1 = self._model[word1] / np.linalg.norm(self._model[word1], 2)
        v2 = self._model[word2] / np.linalg.norm(self._model[word2], 2)

        return np.dot(v1, v2)

    def may_construct_word_vec(self, word):

        return word in self.get_vocab()




