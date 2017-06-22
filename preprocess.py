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
import nltk
import sys
from nltk.corpus import stopwords


def setup():
    # setup needed libraries, directories etc. TODO directories
    nltk.download('punkt')

class TokenizerBase():
    def split_to_words(self, s, delimiter='[.,?!:; {}()"\[" "\]"" "\n"]'):
        l = re.split(delimiter, s)
        l = [v for v in l if v != ''] #remove all empty strings
        return l

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


class SimpleGermanTokenizer(TokenizerBase):
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

class NonStemmingTokenizer(TokenizerBase):
    # https://github.com/devmount/GermanWordEmbeddings/blob/master/preprocessing.py
    def tokenize(self, s):
        # punctuation and stopwords
        punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', u'„', '„', u'“', '“', '\'',
                              '[', ']', '{', '}', '?', '!', '-', u'–', '+', '*', '--', '\'\'', '``']

        punctuation = '?.!/;:()&+"\n"'
        # stop_words = [self.replace_umlauts(token) for token in stopwords.words('german')]

        # replace umlauts
        s = self.replace_umlauts(s)
        # get word tokens
        words = nltk.word_tokenize(s)
        # filter punctuation and stopwords
        words = [x for x in words if x not in punctuation_tokens]

        # function to remove all punctuations at the beginning of a word
        def remove_start_punct(word):
            while word and (word[0] in punctuation_tokens):
                word = word[1:]
            return word

        # remove all punctuations at the beginning of a word
        words = [remove_start_punct(x) for x in words]

        # remove all undesired punctuations at any location
        words = [re.sub('[' + punctuation + ']', '', x) for x in words]

        # process words
        words = [x.lower() for x in words]

        # remove stopwords TODO activate maybe
        # words = [x for x in words if x not in stop_words]

        return words


class SentenceExtractor():
    # not used so far
    # idea from https://github.com/devmount/GermanWordEmbeddings/blob/master/preprocessing.py
    def extract_sentences(self, s):
        # sentence detector
        sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
        sentences = sentence_detector.tokenize(s)

        return sentences


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

    return tokens


def tokens_from_dir(directory, tokenizer):
    """
    creates
    - tokenSet: a set of tokens using all *.txt files of any subdirectory of 'directory'
    - tokenList: a list-of-lists containing all files as tokenized strings (each item is a list of all tokens found in one file)
    """
    print("Read and tokenize data from directory '", directory, "'")
    tokenSet = set()
    tokenList = []

    n = 0
    # iterate over all .txt files
    for dirpath, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.txt'):
            n += 1
            sys.stdout.write( "Reading File "+ str(n) + '\r')

            with open(os.path.join(dirpath, filename), 'r') as file:
                # create tokens from each file
                tokens = get_tokens_from_file(file, tokenizer)
                tokenList.append(tokens)
                tokenSet |= set(tokens)
                file.close()

    return tokenSet, tokenList


def get_tokenizer(config):
    tk = config.config['tokenizer']

    if tk == 'sgt':
        tokenizer = SimpleGermanTokenizer()
    elif tk == 'nst':
        tokenizer = NonStemmingTokenizer()
    else:
        # Default
        print("Warining: Couldn't find specified tokenizer. Continuing with default tokenizer. ")
        tokenizer = NonStemmingTokenizer()

    return tokenizer


def create_train_data(train_data_src, raw_data_dir, config):

    print("Creating new training data. ")

    ### create needed directories TODO

    ### Create needed token-datastructures
    tokenizer = get_tokenizer(config)

    print("Using this Tokenizer: ", str(tokenizer.__class__).split('.')[1].split("'")[0])

    # - tokenSet: each token appears only once
    # - tokenList: entire data of stemmed tokens, each item is a list of all tokens of an article
    tokenSet, tokenList = tokens_from_dir(raw_data_dir, tokenizer)

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
    train_file = open(train_data_src, 'w+')

    for item in article_list:
        train_file.write("%s\n" % item)

    train_file.close()

