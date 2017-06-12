import word2vec as w2v
import numpy as np
import preprocess as pp
import os

VAL_DIR = 'data/validation_data/'
VAL_FILE = 'german_doesntfit1.txt'



def validate_model(model_fp):
    print("Start validation. Loading model.")
    model = w2v.load(model_fp)

    val_file_path = os.path.join(VAL_DIR, VAL_FILE)
    test_doesntfit(model, val_file_path)

    return model #TODO remove return value, only used for development


def vec_dot(model, w1, w2):
    return np.dot(model[w1], model[w2])


def doesntfit(model, word_list):
    """
    - compares each word-vector to mean of all word-vectors of word_list using the vector dot-product
    - vector with lowest dot-produt to mean-vector is regarded as the one that dosen't fit

    """
    used_words = [word for word in word_list if word in model]
    n_used_words = len(used_words)
    n_words = len(word_list)

    if n_used_words != n_words:
        ignored_words = set(word_list) - set(used_words)
        print("vectors for words %s are not present in the model, ignoring these words: ", ignored_words)
    if not used_words:
        print("cannot select a word from an empty list.")

    vectors = np.vstack(model.get_vector(word) for word in used_words)
    mean = np.mean(vectors, axis=0)
    dists = np.dot(vectors, mean)

    return sorted(zip(dists, used_words))[0][1]

def test_doesntfit(model, filepath):
    """
    - tests all doesntfit-questions (lines) of file
    - a doesnt-fit question is of the format "word_1 word_2 ... word_N word_NotFitting"
        where word_1 to word_n are members of a category but word_NotFitting isn't

        eg. "Auto Motorrad Fahrrad Ampel"

    """
    print("Validating 'doesntfit' with file", filepath)

    num_lines = sum(1 for line in open(filepath))
    num_questions = 0
    num_right = 0
    sgt = pp.SimpleGermanTokenizer()


    # get questions
    with open(filepath) as f:
        questions = f.read().splitlines()
        tk_questions = [sgt.tokenize(q) for q in questions]

    # test each question
    for question in tk_questions:

        # check if all words exist in vocabulary
        if all(word in model for word in question):
            num_questions += 1
            if doesntfit(model, question) == question[-1]:
                num_right += 1

    # calculate result
    correct_matches = np.round(num_right/float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = np.round(num_questions/float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    print('Doesn\'t fit correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    print('Doesn\'t fit coverage: {0}% ({1}/{2})'.format(str(coverage), str(num_questions), str(num_lines)))





def visualize_word(model, word):
    {
        # TODO
    }




