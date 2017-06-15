import word2vec
import multiprocessing

from subprocess import call


def make_emb_from_file(train_data_src, emb_model_src, config):

    ### embedding parameters

    # embedding vector dimension
    emb_dim = config.config['embedding_vector_dim']

    # minimum number a token has to appear to be included in model
    min_count = config.config['min_token_appearance']

    # number of cores
    n_cores = multiprocessing.cpu_count()

    print("Start training the model.")
    word2vec.word2vec(train_data_src, emb_model_src, size=emb_dim, window=5, sample='1e-5', hs=0,
                      negative=5, threads=n_cores, iter_=5, min_count=min_count, alpha=0.025,
                      debug=2, binary=1, cbow=1, save_vocab=None, read_vocab=None,
                      verbose=False)

    print("Training finsihed. \nModel saved at:", emb_model_src)

    """
    word2vec execution
    Parameters for training:
        train <file_path>
            Use text data from <file_path> to train the model
        output <file_path>
            Use <file_path> to save the resulting word vectors / word clusters
        size <int>
            Set size of word vectors; default is 100
        window <int>
            Set max skip length between words; default is 5
        sample <float>
            Set threshold for occurrence of words. Those that appear with
            higher frequency in the training data will be randomly
            down-sampled; default is 0 (off), useful value is 1e-5
        hs <int>
            Use Hierarchical Softmax; default is 1 (0 = not used)
        negative <int>
            Number of negative examples; default is 0, common values are 5 - 10
            (0 = not used)
        threads <int>
            Use <int> threads (default 1)
        iter_ = number of iterations (epochs) over the corpus. Default is 5.
        min_count <int>
            This will discard words that appear less than <int> times; default
            is 5
        alpha <float>
            Set the starting learning rate; default is 0.025
        debug <int>
            Set the debug mode (default = 2 = more info during training)
        binary <int>
            Save the resulting vectors in binary moded; default is 0 (off)
        cbow <int>
            Use the continuous bag of words model; default is 1 (skip-gram
            model)
        save_vocab <file>
            The vocabulary will be saved to <file>
        read_vocab <file>
            The vocabulary will be read from <file>, not constructed from the
            training data
        verbose
            Print output from training
    """
