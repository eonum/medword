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
import chardet


def setup():
    # setup needed libraries, directories etc. TODO construct need directories automatically
    nltk.download('punkt')

class TokenizerBase():
    def split_to_words(self, s, delimiter='[.,?!:; {}()"\[" "\]"" "\n"]'):
        l = re.split(delimiter, s)
        l = [v for v in l if v != ''] #remove all empty strings
        return l

    def replace_umlauts(self, text):
        res = text
        return res

    def replace_special_chars(self, text):
        res = text
        res = res.replace(u'ß', 'ss')
        res = res.replace(u'—', '-')
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

        # Define at which chars you want to split words
        # split_chars = ['-', '/', '\\\\', '+', '|']
        split_chars = ['/', '\\\\', '+', '|']
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

        s = self.replace_special_chars(s)

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

        # remove everything except
        words = [re.sub(r'[^a-z0-9%ÜÖÄÉÈÀéèàöäü=><†@≥≤\s\-\/]', '', x) for x in words]

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

    data = file.read()
    tokens = tokenizer.tokenize(data)

    return tokens


def tokens_from_dir(directory, tokenizer, train_file=None, valid_tokens=None):
    """
    - creates tokens using all *.txt files of any subdirectory of 'directory'
    - if train_file is specified, store all tokens in train_file
    - if valid_tokens is specified, only keep the tokens that occur in
      valid_tokens
    """
    print("Read and tokenize data from directory '", directory, "'")
    tokenSet = set()
    total_tokens = 0
    n_files = 0
    n_bad_encoding = 0
    # iterate over all .txt files
    for dirpath, dirs, files in os.walk(directory, followlinks=True):
        for filename in fnmatch.filter(files, '*.txt'):
            n_files += 1
            sys.stdout.write( "Reading File "+ str(n_files) + '\r')

            #try to find the encoding
            encodingInfo = chardet.detect(open(os.path.join(dirpath, filename),
                                   "rb").read())

            # skip the file, if the encoding is unknown
            encoding = encodingInfo['encoding'];
            if (not encoding or encodingInfo['confidence'] < 0.8):
                n_bad_encoding +=1
                continue


            with open(os.path.join(dirpath, filename), 'r', encoding=encoding) \
                    as file:
                # create tokens from each file
                tokens = get_tokens_from_file(file, tokenizer)

                if(valid_tokens is not None):
                    # remove all tokens, which are not in valid_tokens
                    tokens = [t for t in tokens if t in valid_tokens]

                if (train_file is not None):
                    # append token_string to train_file
                    token_string = " ".join(tokens) + " \n"

                    # replace multiple whitespaces with a single one
                    token_string = re.sub('\s+', ' ', token_string)

                    # save in utf-8 format
                    # TODO remove illegal non-utf-8 symbols
                    # (read and write should decode and encode in utf-8 by standard in python3,
                    # the once appeared error could not be reconstructed)
                    train_file.write(token_string)

                # build set of all tokens and count total number of found tokens
                tokenSet |= set(tokens)
                total_tokens += len(tokens)

                file.close()

    n_good_files = n_files - n_bad_encoding
    print("Found %d different tokens in %d articles, total training size: "
          "%d tokens." % (len(tokenSet), n_good_files , total_tokens))
    print("%d files could not be decoded." % n_bad_encoding)

    return tokenSet;


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

    ### Create needed token-datastructures
    tokenizer = get_tokenizer(config)

    print("Using this Tokenizer: ", str(tokenizer.__class__).split('.')[1].split("'")[0])

    # open training data file
    train_file = open(train_data_src, 'w+')

    # Create tokens from raw_data_dir and store them in train_file
    tokens_from_dir(raw_data_dir, tokenizer, train_file)

    # close training date file
    train_file.close()


def create_intersection_train_data(train_data_src, train_data_dir, config):
    # remove irregular tokens from pdf-data and webcrawler-data:
    #
    # 1) generate vocab set for pdf-data, webcrawler-data, wiki_dumps
    #    independently
    # 2) keep only vocab, which occurs in all sets (intersection-set)
    # 3) when creating the train data, remove words which are not in
    #    intersection set by a TBD-strategy (e.g. remove only word, remove line,
    #    remove sentence ...)

    # Verification of relevance:
    # - check words which are not in intersection manually
    # - ...
    raw_data_dir = os.path.join(train_data_dir, 'raw_data/')

    # assumes the following sub-folders
    pdf_folder = os.path.join(raw_data_dir, 'medical_books_plaintxt/')
    crawler_folder = os.path.join(raw_data_dir, 'medtextcollector_output/')
    wiki_folder = os.path.join(raw_data_dir, 'wiki_dumps_txts/')
    codes_folder = os.path.join(raw_data_dir, 'ICD_CHOP/')

    print("Creating new intersection training data. ")

    ### Create needed token-datastructures
    tokenizer = get_tokenizer(config)

    print("Using this Tokenizer: ",
          str(tokenizer.__class__).split('.')[1].split("'")[0])

    # Create tokens from raw_data_dir and store them in train_file
    pdf_token_set = tokens_from_dir(pdf_folder, tokenizer)
    crawler_token_set = tokens_from_dir(crawler_folder, tokenizer)
    wiki_token_set = tokens_from_dir(wiki_folder, tokenizer)

    # compute intersection
    intersection_token_set = pdf_token_set & crawler_token_set & wiki_token_set


    ### DEBUG & INSPECTION ###
    # compute remaining parts (for debug / inspection purpose)
    remain_pdf_token_set = pdf_token_set - intersection_token_set
    remain_crawler_token_set = crawler_token_set - intersection_token_set
    remain_wiki_token_set = wiki_token_set - intersection_token_set

    # save sets for inspection
    file_names = ['pdf_set', 'crawler_set', 'wiki_set', 'inter_set',
                  'pdf_excl', 'crawler_excl', 'wiki_excl']
    for i, data_set in enumerate([pdf_token_set, crawler_token_set,
                                  wiki_token_set, intersection_token_set,
                                  remain_pdf_token_set,
                                  remain_crawler_token_set,
                                  remain_crawler_token_set]):

        out_src = os.path.join(train_data_dir, 'processed_data/' +
                               file_names[i] + '.txt')
        with open(out_src, 'w') as file:
            file.writelines([item + '\n' for item in data_set])

    ### END DEBUG & INSPECTION ###

    # open training data file
    train_file = open(train_data_src, 'w+')

    # Create tokens from multiple dirs and append them to train_file
    # only tokens included in valid_tokens are kept
    directories = [pdf_folder, crawler_folder, wiki_folder]
    for dir in directories:
        tokens_from_dir(dir, tokenizer, train_file,
                    valid_tokens=intersection_token_set)

    # also add CHOP and ICD tokens to train_file but keep all (not only the ones
    # that are in intersection_token_set)
    tokens_from_dir(codes_folder, tokenizer, train_file, valid_tokens=None)


    # close training date file
    train_file.close()


    ### DEBUG & INSPECTION ###
    return pdf_token_set, crawler_token_set, wiki_token_set, \
           intersection_token_set, remain_pdf_token_set, \
           remain_crawler_token_set, remain_wiki_token_set
    ### END DEBUG & INSPECTION ###
