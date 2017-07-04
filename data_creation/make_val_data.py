"""
This script contains functions to create validation data.
It uses *.csv files produced by the script 'extract_synonyms.py' as input

"""
import pandas
import numpy as np
import os

DATADIR = 'data/validation_data/'
syn_raw_fn = 'raw_data/synonyms_v2.csv'
syn_output_fn = 'german_synonyms_phrases.txt'


def make_synonyms(input_src, output_src):
    """
    ... writes each synonym pair (eg. [Gatte Ehemann]) on a single line of the output *.txt file.
    - If there are multiple synonyms in one synonym-set (eg. Gatte, Ehemann, Ehepartner)
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
    synonyms = []

    for synonyms_set in all_synonyms:

        if (len(synonyms_set) >= 2 ):
            # Found at least one valid synonym pair
            for i in range(len(synonyms_set)):
                for j in range(i+1, len(synonyms_set)):
                    synonyms.append(synonyms_set[i] + ";" + synonyms_set[j])
        

    with open(output_src, 'w+') as f:
        f.writelines(["%s\n" % item for item in synonyms])



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