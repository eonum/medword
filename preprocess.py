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

DATA_DIR = '/Users/Fabian/Developer/eonum/Coding/medword/data/wiki_data/'
#DATA_DIR = '/Users/Fabian/Developer/eonum/Coding/medword/data/data_test/'


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


def get_tokens_from_file(file, tokenizer):
    file.seek(0)  # reset file iterator
    data = file.read().replace('\n', '')
    tokens = tokenizer.tokenize(data)
    #print(tokens)
    return tokens



def tokens_from_dir(directory):
    """
    creates
    - a set of tokens using all *.txt files of any subdirectory of 'directory'
    - a list containing all files as tokenized strings
    """
    print("Making tokenSet from directory '", directory, "'")
    tokenSet = set()
    tokenList = []
    sgt = SimpleGermanTokenizer()

    # iterate over all .txt files
    for dirpath, dirs, files in os.walk(directory):
        for filename in fnmatch.filter(files, '*.txt'):
            with open(os.path.join(dirpath, filename), 'r') as file:
                # create tokens from each file
                tokens = get_tokens_from_file(file, sgt)
                tokenList.append(tokens)
                tokenSet |= set(tokens)
                file.close()

    return tokenSet, tokenList



def create_token_datastructs():

    tokenSet, tokenList = tokens_from_dir(DATA_DIR)
    n_tokens = len(tokenSet)

    # Create look-up structures to get token-index from from token
    # (used to find index of embedding vector for given token)

    # look-up dictionary [token -> index]
    token_indx_dic = {}
    # look-up list [index -> token]
    indx_token_list = []

    i = 0
    for t in sorted(tokenSet): # maybe sort if needed
        token_indx_dic[t] = i
        indx_token_list.append(t)
        i += 1

    # TODO maybe remove token_indx_dic, and indx_token_list as not used so far
    return tokenSet, tokenList, token_indx_dic, indx_token_list



def create_train_data(train_fp):
    # Create needed token-datastructures:
    # - tokenSet: each token appears only once
    # - tokenList: entire data of stemmed tokens, each item is a list of all tokens of an article
    print("Creating new training data. ")

    tokenSet, tokenList, _, _ = create_token_datastructs()
    total_tokens = sum([len(item) for item in tokenList])

    print("Found %d different tokens in %d articles, total training size: %d tokens"
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

