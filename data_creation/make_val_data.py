"""
This script contains functions to create validation data.
It uses *.csv files produced by the script 'extract_synonyms.py' as input

"""
import pandas
import numpy as np
import os

DATADIR = 'data/validation_data/'
syn_raw_fn = 'raw_data/synonyms_v2.csv'
syn_output_fn = 'german_synonyms3.txt'


def make_synonyms(input_src, output_src):
    """
    ... writes each synonym pair (eg. [Gatte Ehemann]) on a single line of the output *.txt file.
    - Only one-word synonyms are considered (no phrases like 'verheirateter Mann').
    - If there are multiple one-word synonyms in one synonym-set (eg. Gatte, Ehemann, Ehepartner)
    then each combination ist taken (eg. [Gatte Ehemann], [Gatte Ehepartner], [Ehemann Ehepartner])
    :param input_src: csv_file with variable line-length
    :param output_src: empty *.txt file

    """
    with open(input_src) as f:
        lines = f.read().splitlines()
        n_lines = len(lines)

    # list of all synonym sets
    all_synonyms = [list(filter(None, syn.split(';'))) for syn in lines ]

    # list of all one-word synonym pairs
    single_word_synonyms = []

    for synonyms_set in all_synonyms:

        # split all phrases in a synonym set to check the number of words in that phrase
        syn_phrase_split = [syn_phrase.split() for syn_phrase in synonyms_set]
        numb_words_per_phrase = [len(phrase) for phrase in syn_phrase_split]

        single_words = [syn_phrase_split[indx][0] for (indx, syn) in enumerate(numb_words_per_phrase) if syn == 1]

        if (len(single_words) >= 2 ):
            # Found at least one valid synonym pair
            for i in range(len(single_words)):
                for j in range(i+1, len(single_words)):
                    single_word_synonyms.append(single_words[i] + " " + single_words[j])
        

    with open(output_src, 'w+') as f:
        f.writelines(["%s\n" % item for item in single_word_synonyms])



def make_validation_data():
    """
    ... loads input and output source names and calls make_synonyms on them.
    """
    # source paths for files
    syn_raw_src = os.path.join(DATADIR, syn_raw_fn)
    syn_output_src = os.path.join(DATADIR, syn_output_fn)

    # make synonym file (*.txt)
    make_synonyms(syn_raw_src, syn_output_src)


if __name__ == "__main__":
    make_validation_data()