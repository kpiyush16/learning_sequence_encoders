import bcolz
import pickle
import numpy as np


def glove_emb(target_vocab, glove_path = ".", emb_dim = 100):
    vectors = bcolz.open(f'{glove_path}/6B.100.dat')[:]
    words = pickle.load(open(f'{glove_path}/6B.100_words.pkl', 'rb'))
    word2idx = pickle.load(open(f'{glove_path}/6B.100_idx.pkl', 'rb'))

    glove = {w: vectors[word2idx[w]] for w in words}

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, 100))
    words_found = 0

    for i, word in enumerate(target_vocab):
        try: 
            weights_matrix[i] = glove[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))