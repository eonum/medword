import numpy as np
import preprocess as pp
import os
from random import randint
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def validate_model(embedding, emb_model_dir, emb_model_fn):
    print("Start validation. Loading model. \n")

    # load config
    config = embedding.config

    # load model
    embedding.load_model(emb_model_dir, emb_model_fn)

    # directories and filenames
    val_dir = config.config['val_data_dir']

    doesntfit_fn = config.config['doesntfit_file']
    doesntfit_src = os.path.join(val_dir, doesntfit_fn)

    synonyms_fn = config.config['synonyms_file']
    syn_file_src = os.path.join(val_dir, synonyms_fn)


    # test with doesn't fit questions
    test_doesntfit(embedding, doesntfit_src)

    # test with synonyms
    test_synonyms(embedding, syn_file_src)


#### Doesn't Fit Validation ####


def doesntfit(embedding, word_list):
    """
    - compares each word-vector to mean of all word-vectors of word_list using the vector dot-product
    - vector with lowest dot-produt to mean-vector is regarded as the one that dosen't fit

    """
    used_words = [word for word in word_list if embedding.may_construct_word_vec(word)]
    n_used_words = len(used_words)
    n_words = len(word_list)

    if n_used_words != n_words:
        ignored_words = set(word_list) - set(used_words)
        print("vectors for words %s are not present in the model, ignoring these words: ", ignored_words)
    if not used_words:
        print("cannot select a word from an empty list.")

    vectors = np.vstack(embedding.word_vec(word) for word in used_words)
    mean = np.mean(vectors, axis=0)
    dists = np.dot(vectors, mean)

    return sorted(zip(dists, used_words))[0][1]

def test_doesntfit(embedding, file_src):
    """
    - tests all doesntfit-questions (lines) of file
    - a doesnt-fit question is of the format "word_1 word_2 ... word_N word_NotFitting"
        where word_1 to word_n are members of a category but word_NotFitting isn't

        eg. "Auto Motorrad Fahrrad Ampel"

    """
    # load config
    config = embedding.config

    print("Validating 'doesntfit' with file", file_src)

    num_lines = sum(1 for line in open(file_src))
    num_questions = 0
    num_right = 0

    tokenizer = pp.get_tokenizer(config)


    # get questions
    with open(file_src) as f:
        questions = f.read().splitlines()
        tk_questions = [tokenizer.tokenize(q) for q in questions]

    # TODO: check if tokenizer has splitted one word to mulitple words and handle it.
    # So far no word in the doesnt_fit testfile should be splitted

    # vocab used to speed checking if word is in vocabulary
    # (also checked by embedding.may_construct_word_vec(word))
    vocab = embedding.get_vocab()
    # test each question
    for question in tk_questions:

        # check if all words exist in vocabulary
        if all(((word in vocab) or (embedding.may_construct_word_vec(word))) for word in question):
            num_questions += 1
            if doesntfit(embedding, question) == question[-1]:
                num_right += 1

    # calculate result
    correct_matches = np.round(num_right/np.float(num_questions)*100, 1) if num_questions>0 else 0.0
    coverage = np.round(num_questions/np.float(num_lines)*100, 1) if num_lines>0 else 0.0
    # log result
    print("\n*** Doesn't fit ***")
    print('Doesn\'t fit correct:  {0}% ({1}/{2})'.format(str(correct_matches), str(num_right), str(num_questions)))
    print('Doesn\'t fit coverage: {0}% ({1}/{2}) \n'.format(str(coverage), str(num_questions), str(num_lines)))



#### Synonyms Validation ####
def test_synonyms(embedding, file_src):
    """
    - tests all synonym-questions (lines) of file
    - a synonym-question is of the format "word_1 word_2"
        where word_1 and word_2 are synonyms

        eg. "Blutgerinnsel Thrombus"
    - for word_1 check if it appears in the n closest words of word_2 using "model.cosine(word, n)"
        and vice-versa
    - for each synonym-pair TWO CHECKS are made therefore (non-symmetric problem)

    """
    print("Validating 'synonyms' with file", file_src)

    config = embedding.config

    num_lines = sum(1 for line in open(file_src))
    num_questions = 0
    cos_sim_sum_synonyms = 0

    tokenizer = pp.get_tokenizer(config)

    # get questions which are still of lenght 2 after tokenization
    # TODO: improve for compound words (aaa-bbb) which are splitted by the tokenizer
    tk_questions = []
    with open(file_src, 'r') as f:
        questions = f.read().splitlines()

        for q in questions:
            synonyms = q.split(';')#tokenizer.tokenize(q)
            synonyms = [" ".join(tokenizer.tokenize(synonym)) for synonym in synonyms]
            if len(synonyms) == 2:
                tk_questions.append(synonyms)

    vocab = embedding.get_vocab()

    # test each question
    for tk_quest in tk_questions:

        # check if all words exist in vocabulary
        if all(((word in vocab) or embedding.may_construct_word_vec(word))  for word in tk_quest):
            num_questions += 1

            w1 = tk_quest[0]
            w2 = tk_quest[1]

            cos_sim_sum_synonyms += embedding.similarity(w1, w2)


    # compute avg cosine similarity for random vectors to relate to avg_cosine_similarity of synonyms
    vocab_size = len(vocab)
    n_vals = 1000
    similarity_sum_rand_vec = 0
    vals1 = [randint(0, vocab_size -1) for i in range(n_vals)]
    vals2 = [randint(0, vocab_size -1) for i in range(n_vals)]

    for v1, v2 in zip(vals1, vals2):
        similarity_sum_rand_vec += embedding.similarity(vocab[v1], vocab[v2])

    avg_cosine_similarity_rand_vec = similarity_sum_rand_vec / np.float(n_vals)


    # calculate result
    avg_cosine_similarity_synonyms = (cos_sim_sum_synonyms / num_questions) if num_questions>0 else 0.0
    coverage = np.round(num_questions/np.float(num_lines)*100, 1) if num_lines>0 else 0.0

    # log result
    print("\n*** Cosine-Similarity ***")
    print("Synonyms avg-cos-similarity (SACS):", avg_cosine_similarity_synonyms, "\nRandom avg-cos-similarity (RACS):", avg_cosine_similarity_rand_vec,
          "\nRatio SACS/RACS:", avg_cosine_similarity_synonyms/float(avg_cosine_similarity_rand_vec))
    print("\n*** Word Coverage ***")
    print("Synonyms: {0} pairs in input. {1} pairs after tokenization. {2} pairs in model-vocabulary.".format(str(num_lines), str(len(tk_questions)), str(num_questions)))

    print("Synonyms coverage: {0}% ({1}/{2})\n".format(str(coverage), str(2*num_questions), str(2*num_lines), ))


#### Visualization ####
def visualize_words(embedding, word_list, n_nearest_neighbours):

    # get indexes and words that you want to visualize
    words_to_visualize = []
    # word_indexes_to_visualize = []

    # get all words and neighbors that you want to visualize
    for word in word_list:
        if not embedding.may_construct_word_vec(word):
            continue
        words_to_visualize.append(word)
        # word_indexes_to_visualize.append(model.ix(word))

        # get neighbours of word

        neighbours = [n for (n, m) in embedding.most_similar_n(word, n_nearest_neighbours)]

        words_to_visualize.extend(neighbours)
        #word_indexes_to_visualize.extend(indexes)


    # get vectors from indexes to visualize
    if words_to_visualize == []:
        print("No word found to show.")
        return

    emb_vectors = np.vstack([embedding.word_vec(word) for word in words_to_visualize])


    # project down to 2D
    pca = PCA(n_components=2)
    emb_vec_2D = pca.fit_transform(emb_vectors)

    n_inputs = len(word_list)

    for i in range(n_inputs):

        # group word and it's neighbours together (results in different color in plot)
        lower = i*n_nearest_neighbours + i
        upper = (i+1)*n_nearest_neighbours + (i+1)

        # plot 2D
        plt.scatter(emb_vec_2D[lower:upper, 0], emb_vec_2D[lower:upper, 1])
        for label, x, y in zip(words_to_visualize, emb_vec_2D[:, 0], emb_vec_2D[:, 1]):
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    # find nice axes for plot
    lower_x = min(emb_vec_2D[:, 0])
    upper_x = max(emb_vec_2D[:, 0])
    lower_y = min(emb_vec_2D[:, 1])
    upper_y = max(emb_vec_2D[:, 1])

    # 10% of padding on all sides
    pad_x = 0.1 * abs(upper_x - lower_x)
    pad_y = 0.1 * abs(upper_y - lower_y)

    plt.xlim([lower_x - pad_x, upper_x + pad_x])
    plt.ylim([lower_y - pad_y, upper_y + pad_y])

    plt.show()







