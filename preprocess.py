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
        punctuation_tokens = ['.', '..', '...', ',', ';', ':', '"', u'„', '„', u'“', '“', '\'',
                              '[', ']', '{', '}', '(', ')', '<', '>', '?', '!', '-', u'–', '+',
                              '*', '--', '\\', '\'\'', '``', '‚', '‘', '\n', '\\n', '']

        punctuation = ['?', '.', '!', '/', ';', ':', '(', ')', '&', '\n']
        split_chars = ['-', '/', '\\\\', '+', '|']

        # stop_words = [self.replace_umlauts(token) for token in stopwords.words('german')]


        # replace umlauts
        s = self.replace_umlauts(s)

        # replace newline chars
        def remove_newlines(document):
            document = re.sub('\\n', ' ', document)
            document = re.sub('\\\\n', ' ', document)
            document = re.sub('\n', ' ', document)

            return document

        s = remove_newlines(s)

        # get word tokens
        words = nltk.word_tokenize(s)


        # filter punctuation tokens
        words = [x for x in words if x not in punctuation_tokens]

        # remove stopwords
        # words = [x for x in words if x not in stop_words]

        # split words at defined characters
        delimiters = '[' + "".join(split_chars) + ']'

        flat_words = []
        for x in words:
            flat_words.extend(re.split(delimiters, x))

        words = flat_words


        # functions to remove all punctuations at the beginning and end of a word
        # (in case something in the nltk.word_tokenize() was left over)
        def remove_start_punct(word):
            while word and (word[0] in punctuation_tokens):
                word = word[1:]
            return word

        def remove_end_puntc(word):
            while word and (word[-1] in punctuation_tokens):
                word = word[:-1]
            return word

        # remove all punctuations at the beginning and ending of a word
        words = [remove_start_punct(x) for x in words]
        words = [remove_end_puntc(x) for x in words]

        # remove all undesired punctuations at any location
        words = [re.sub('[' + "".join(punctuation) + ']', '', x) for x in words]

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


def tokens_from_dir(directory, tokenizer, train_file):
    """
    - creates tokens using all *.txt files of any subdirectory of 'directory'
    - stores them in train_file
    """
    print("Read and tokenize data from directory '", directory, "'")
    tokenSet = set()
    total_tokens = 0
    n_files = 0
    # iterate over all .txt files
    for dirpath, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.txt'):
            n_files += 1
            sys.stdout.write( "Reading File "+ str(n_files) + '\r')

            with open(os.path.join(dirpath, filename), 'r') as file:
                # create tokens from each file
                tokens = get_tokens_from_file(file, tokenizer)

                # append token_string to train_file
                token_string = " ".join(tokens) + " \n"

                # replace multiple whitespaces with a single one
                token_string = re.sub('\s+', ' ', token_string)

                # save in utf-8 format
                #token_string = bytes(token_string).decode('utf-8','ignore')
                train_file.write(token_string)

                # build set of all tokens and count total number of found tokens
                tokenSet |= set(tokens)
                total_tokens += len(tokens)

                file.close()


    print("Found %d different tokens in %d articles, total training size: %d tokens."
          % (len(tokenSet), n_files, total_tokens))


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

    # open training data file
    train_file = open(train_data_src, 'w+')

    # Create tokens from raw_data_dir and store them in train_file
    tokens_from_dir(raw_data_dir, tokenizer, train_file)

    # close training date file
    train_file.close()

