# -*- coding: utf-8 -*-

"""
This file should contain functions for the following operations:
- load data from raw txt format (in German)
- stemming
- (maybe) remove stop-words (e.g. der, die, das, in, am ...)
- create dictionary containing all desired words

"""
import os
import fnmatch
import re
from nltk.stem.snowball import GermanStemmer
import nltk.data
from nltk.corpus import stopwords
import itertools

# DATA_DIR = 'data/train_data/'
DATA_DIR = 'data/data_small/'


class SimpleTokenizer():
    def split_to_words(self, s, delimiter='[.,?!:; ]'):
        l = re.split(delimiter, s)
        l = [v for v in l if v != ''] #remove all empty strings
        return l


class SimpleGermanTokenizer(SimpleTokenizer):
    def tokenize(self, s):
        words = self.split_to_words(s)
        stemmed_words = self.stem_words(words)
        return stemmed_words

    def stem_words(self, words):
        stemmer = GermanStemmer()
        stemmed_words = []
        for word in words:
            stemmed_words.append(stemmer.stem(word))
        return stemmed_words

class GWETokenizer():
    # https://github.com/devmount/GermanWordEmbeddings/blob/master/preprocessing.py
    def tokenize(self, s):
        # punctuation and stopwords
        punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-',
                              u'–', '+', '*', '--', '\'\'', '``']
        punctuation = '?.!/;:()&+'
        stop_words = [self.replace_umlauts(token) for token in stopwords.words('german')]

        # sentence detector
        sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
        sentences = sentence_detector.tokenize(s)

        output = []
        for sentence in sentences:
            # replace umlauts
            sentence = self.replace_umlauts(sentence)
            # get word tokens
            words = nltk.word_tokenize(sentence)
            # filter punctuation and stopwords
            words = [x for x in words if x not in punctuation_tokens]
            words = [re.sub('[' + punctuation + ']', '', x) for x in words]
            #words = [x for x in words if x not in stop_words]
            output.append(words)

        return list(itertools.chain.from_iterable(output))

    def replace_umlauts(self, text):
        res = text
        res = res.replace(u'ä', 'ae')
        res = res.replace(u'ö', 'oe')
        res = res.replace(u'ü', 'ue')
        res = res.replace(u'Ä', 'Ae')
        res = res.replace(u'Ö', 'Oe')
        res = res.replace(u'Ü', 'Ue')
        res = res.replace(u'ß', 'ss')
        return res


def make_directory(base_directory, new_subdirectory):
    new_subdir_path = os.path.join(base_directory, new_subdirectory)
    if not os.path.exists(new_subdir_path):
        os.makedirs(new_subdir_path)


def get_tokens_from_file(file, tokenizer):
    """
    reads file and returns a list of all ovserved tokens (stemmed)
    """
    file.seek(0)  # reset file iterator
    data = file.read().replace('\n', '')
    tokens = tokenizer.tokenize(data)
    print(str(file))
    print(tokens)
    return tokens


def tokens_from_dir(directory, tokenizer):
    """
    creates
    - tokenSet: a set of tokens using all *.txt files of any subdirectory of 'directory'
    - tokenList: a list-of-lists containing all files as tokenized strings (each item is a list of all tokens found in one file)
    """
    print("Making tokenSet from directory '", directory, "'")
    tokenSet = set()
    tokenList = []


    # iterate over all .txt files
    for dirpath, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.txt'):
            with open(os.path.join(dirpath, filename), 'r') as file:
                # create tokens from each file
                tokens = get_tokens_from_file(file, tokenizer)
                tokenList.append(tokens)
                tokenSet |= set(tokens)
                file.close()

    return tokenSet, tokenList


def create_train_data(train_fp):

    print("Creating new training data. ")

    # create needed directories TODO

    # Create needed token-datastructures
    # - tokenSet: each token appears only once
    # - tokenList: entire data of stemmed tokens, each item is a list of all tokens of an article
    sgt = SimpleGermanTokenizer()
    gwe = GWETokenizer()

    tokenSet, tokenList = tokens_from_dir(DATA_DIR, gwe)

    total_tokens = sum([len(item) for item in tokenList])

    print("Found %d different tokens in %d articles, total training size: %d tokens."
          % (len(tokenSet), len(tokenList), total_tokens))

    # Create training-file from tokenList.
    # Each item of article_list should contain an entire article.
    # Tokens are separated with a single whitespace ' '.
    article_list = []
    for l in tokenList:
        a = ' '.join(l)
        article_list.append(a)

    # safe training data to file
    train_file = open(train_fp, 'w')

    for item in article_list:
        train_file.write("%s\n" % item)

    train_file.close()

