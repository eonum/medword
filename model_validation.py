import word2vec as w2v
import numpy as np
import preprocess as pp
import os
from random import randint

# data names and paths
VAL_DIR = 'data/validation_data/'
DF_FILE = 'german_doesntfit1.txt'
SYN_FILE = 'german_synonymes3.txt'


def validate_model(model_fp):
    print("Start validation. Loading model. \n")

    # load model
    model = w2v.load(model_fp)

    # test with doesn't fit questions
    val_file_path = os.path.join(VAL_DIR, DF_FILE)
    test_doesntfit(model, val_file_path)

    # test with synonymes
    syn_file_path = os.path.join(VAL_DIR, SYN_FILE)
    test_synonymes(model, syn_file_path, 40)

    return model #TODO remove return value, only used for development


#### Doesn't Fit Validation ####

def cosine_similarity(model, w1, w2):
    # note that embedding vecores are already normalized:
    # v1 = model[w1] / np.linalg.norm(model[w1],2)
    # v2 = model[w2] / np.linalg.norm(model[w2],2)

    v1 = model[w1]
    v2 = model[w2]
    return np.dot(v1, v2)


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
    correct_matches = np.round(num_right/np.float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = np.round(num_questions/np.float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    print("\n*** Doesn't fit ***")
    print('Doesn\'t fit correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    print('Doesn\'t fit coverage: {0}% ({1}/{2}) \n'.format(str(coverage), str(num_questions), str(num_lines)))



#### Synonymes Validation ####
def test_synonymes(model, filepath, n_closest_words=10):
    """
    - tests all synonyme-questions (lines) of file
    - a synonyme-question is of the format "word_1 word_2"
        where word_1 and word_2 are synonymes

        eg. "Blutgerinnsel Thrombus"
    - for word_1 check if it appears in the n closest words of word_2 using "model.cosine(word, n)"
        and vice-versa
    - for each synonyme-pair TWO CHECKS are made therefore (non-symmetric problem)

    """
    print("Validating 'synonymes' with file", filepath)

    num_lines = sum(1 for line in open(filepath))
    num_questions = 0
    cos_sim_sum_synonymes = 0
    num_right = 0

    sgt = pp.SimpleGermanTokenizer()

    # get questions
    with open(filepath, 'r') as f:
        questions = f.read().splitlines()
        tk_questions = [sgt.tokenize(q) for q in questions]

    # test each question
    for tk_quest in tk_questions:

        # check if all words exist in vocabulary
        if all(word in model for word in tk_quest):
            num_questions += 1

            w1 = tk_quest[0]
            w2 = tk_quest[1]
            indexes1, metrices1 = model.cosine(w1, n_closest_words)
            indexes2, metrices2 = model.cosine(w2, n_closest_words)

            if w1 in [word for (word, cos_sim) in model.generate_response(indexes2, metrices2)]:
                num_right += 1
                #print(w1, "is in neighbourhood of ", w2)
            if w2 in [word for (word, cos_sim) in model.generate_response(indexes1, metrices1)]:
                num_right += 1
                #print(w2, "is in neighbourhood of ", w1)

            cos_sim_sum_synonymes += cosine_similarity(model, w1, w2)


    # compute avg cosine similarity for random vectors to relate to avg_cosine_similarity of synonymes
    n_vals = 1000
    similarity_sum_rand_vec = 0
    vals1 = [randint(0, model.vocab.size -1) for i in range(n_vals)]
    vals2 = [randint(0, model.vocab.size -1) for i in range(n_vals)]
    for v1, v2 in zip(vals1, vals2):
        similarity_sum_rand_vec += cosine_similarity(model, model.vocab[v1], model.vocab[v2])

    avg_cosine_similarity_rand_vec = similarity_sum_rand_vec / np.float(n_vals)


    # calculate result
    avg_cosine_similarity_synonymes = (cos_sim_sum_synonymes / num_questions) if num_questions>0 else 0.0
    correct_matches = np.round(num_right/(2*np.float(num_questions))*100, 1) if num_questions>0 else 0.0
    coverage = np.round(num_questions/np.float(num_lines)*100, 1) if num_lines>0 else 0.0

    # log result
    print("Synonymes: {0} pairs in input. {1} pairs in model-vocabulary.".format(str(num_lines), str(num_questions)))
    print("\n*** Cosine-Similarity ***")
    print("Synonymes avg-cos-similarity (SACS):", avg_cosine_similarity_synonymes, "\nRandom avg-cos-similarity (RACS):", avg_cosine_similarity_rand_vec,
          "\nRatio SACS/RACS:", avg_cosine_similarity_synonymes/float(avg_cosine_similarity_rand_vec))
    print("\n*** Synonyme Recognition ***")
    print("Synonymes correct:  {0}% ({1}/{2}), checked {3} closest embedding-vectors"
          "checked per word.".format(str(correct_matches), str(num_right), str(2*num_questions), str(n_closest_words)))
    print("Synonymes coverage: {0}% ({1}/{2})\n".format(str(coverage), str(2*num_questions), str(2*num_lines), ))


#### Visualization ####
def visualize_word(model, word):
    {
        # TODO
    }




