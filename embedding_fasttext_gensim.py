"""
embedding model: FastText by Facebook

- using subword information: character n-grams

https://pypi.python.org/pypi/fasttext#enriching-word-vectors-with-subword-information

"""
from gensim.models.wrappers.fasttext import FastText
import os
import json
from embedding_base import EmbeddingBaseAbstract

class EmbeddingFasttextGensim(EmbeddingBaseAbstract):

    def __init__(self, config):
        self._model = None
        self.config = config


    def train_model(self, train_data_src, emb_model_dir, emb_model_fn):
        print("\nEmbedding Algorithm: fastText gensim")

        ft_path = self.config.config['fasttext_source']
        emb_model_src = os.path.join(emb_model_dir, emb_model_fn)

        # train model
        self._model = FastText.train(ft_path=ft_path, corpus_file=train_data_src)

        # save model
        FastText.save(self._model, emb_model_src)

        # save configuration
        filename, ext = os.path.splitext(emb_model_fn)
        config_fn = filename + '_configuration.json'
        config_src = os.path.join(emb_model_dir, config_fn)

        with open(config_src, 'w') as f:
            json.dump(self.config.config, f, indent=4)

        print("Training finsihed. \nModel saved at:", emb_model_src)



    def similarity(self, word1, word2):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.wv.similarity(word1, word2)


    def most_similar_n(self, word, n=10):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.wv.similar_by_word(word, n)


    def load_model(self, emb_model_dir, emb_model_fn):
        input_filename = os.path.join(emb_model_dir, emb_model_fn)
        self._model = FastText.load(input_filename)


    def get_vocab(self):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.wv.index2word


    def word_vec(self, word):
        if not self._model:
            print("Model not defined. Train or load a model.")
            return ReferenceError

        return self._model.wv.word_vec(word)


    def may_construct_word_vec(self, word):

        return self._model.wv.__contains__(word)

