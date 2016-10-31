import numpy as np
import os
import codecs


def read_vocab(wvocab_path, cvocab_path):
    wvocab = dict()
    cvocab = dict()
    with codecs.open(wvocab_path, 'rb', encoding='latin-1') as fp:
        for line in fp:
            word = line.strip()
            if word not in wvocab:
                wvocab[word] = len(wvocab)+1

    with codecs.open(cvocab_path, 'rb', encoding='latin-1') as fp:
        for line in fp:
            char = line.strip('\n')
            if char not in cvocab:
                cvocab[char] = len(cvocab)+1
    return wvocab, cvocab


def read_embeddings(vocab, emb_path, k=300):
    """
      Loads embedings
      """
    with codecs.open(emb_path, "rb") as f:
        header = f.readline()
        embeddings = np.asarray(np.random.uniform(-0.25, 0.25, (len(vocab)+1, k)), dtype='float32')
        for line in f:
            es = line.split()
            word = es[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array([float(e) for e in es[1:]], dtype='float32')

    embeddings[0, :] = 0.
    return embeddings


def read_trec(data_path, wvocab, cvocab):
    data_set = dict()
    wsents = []
    csents = []
    labels = []
    max_wsent_len = 0
    max_csent_len = 0
    with codecs.open(os.path.join(data_path, 'data.txt'), 'r', encoding='latin-1') as fp:
        for line in fp:
            words = line.strip().split()
            if len(words) > max_wsent_len:
                max_wsent_len = len(words)
            wsents.append(words)

            if len(line.strip()) > max_csent_len:
                max_csent_len = len(line.strip())
            csents.append(line.strip())

    with codecs.open(os.path.join(data_path, 'label.txt'), 'r', encoding='latin-1') as fp:
        for line in fp:
            labels.append(int(line.strip())-1)

    np_wsents = np.zeros((len(wsents), max_wsent_len), dtype='int32')
    np_csents = np.zeros((len(csents), max_csent_len), dtype='int32')
    np_labels = np.zeros((len(wsents),), dtype='int32')
    for i in xrange(len(wsents)):
        for j in xrange(len(wsents[i])):
            np_wsents[i, j] = wvocab[wsents[i][j]]

        for j in xrange(len(csents[i])):
            np_csents[i, j] = cvocab[csents[i][j]]

        np_labels[i] = labels[i]

    data_set['wsents'] = wsents
    data_set['csents'] = csents
    data_set['labels'] = labels
    data_set['np_wsents'] = np_wsents
    data_set['np_csents'] = np_csents
    data_set['np_labels'] = np_labels
    return data_set
