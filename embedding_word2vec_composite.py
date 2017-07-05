import numpy as np
import re
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



        words = re.split(r'[-,?!:; {}()"\[" "\]"" "\n"/\\]', word)
        vector = None

        for word in words:
            end = len(word)

            # TODO: Currently this algorithm is greedy.
            # crop at the beginning if nothing has been found.
            # Or replace with dynamic programming / matrix based tokenization

            while word != '' and end > 0:
                if word[:end] in self._model.vocab_hash:
                    if vector is None:
                        vector = self._model.get_vector(word[:end])
                    else:
                        vector = vector + self._model.get_vector(word[:end])
                    word = word[end:]
                    end = len(word)
                else:
                    end -= 1

        if vector is not None:
            vector = vector / np.linalg.norm(vector)

        return vector

    def most_similar_n(self, word, topn=10):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        word_vec = self.word_vec(word)
        metrics = np.dot(self._model.vectors, word_vec.T)
        indexes = np.argsort(metrics)[::-1][1:topn + 1]
        best_metrics = metrics[indexes]

        return self._model.generate_response(indexes, best_metrics).tolist()


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

        v1 = self.word_vec(word1)
        v2 = self.word_vec(word2)

        return np.dot(v1, v2)

    def may_construct_word_vec(self, word):

        return self.word_vec(word) is not None




