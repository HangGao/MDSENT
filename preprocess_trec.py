"""
Pre-processing script for TREC data.

"""
import os
import glob
import random
import codecs


def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def build_vocab(file_paths, wdst_path, cdst_path, lowercase=True):
    wvocab = set()
    cvocab = set()
    for file_path in file_paths:
        with codecs.open(file_path, 'rb', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if lowercase:
                    line = line.lower()
                wvocab |= set(line.split())
                cvocab |= set(line)
    with codecs.open(wdst_path, 'wb', encoding='latin-1') as f:
        for w in sorted(wvocab):
            f.write(w + '\n')
    with codecs.open(cdst_path, 'wb', encoding='latin-1') as f:
        for c in sorted(cvocab):
            f.write(c + '\n')


def split(base):
    train_file = os.path.join(base, 'train_5500.label')
    test_file = os.path.join(base, 'TREC_10.label')
    
    train_dir = os.path.join(base, 'train')
    dev_dir = os.path.join(base, 'dev')
    test_dir = os.path.join(base, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    label_dict = {}
    with codecs.open(train_file, 'rb', encoding='latin-1') as datafile, \
            codecs.open(os.path.join(train_dir, 'data.txt'), 'wb', encoding='latin-1') as tr_file, \
            codecs.open(os.path.join(dev_dir, 'data.txt'), 'wb', encoding='latin-1') as dv_file, \
            codecs.open(os.path.join(train_dir, 'label.txt'), 'wb', encoding='latin-1') as tr_label, \
            codecs.open(os.path.join(dev_dir, 'label.txt'), 'wb', encoding='latin-1') as dv_label:
            for line in datafile:
                es = line.strip().split()
                label, fine_label = es[0].split(':')
                if label not in label_dict:
                    label_dict[label] = len(label_dict)+1
                sent = ' '.join(es[1:])
                if random.random() < 0.1:
                    dv_file.write(sent + '\n')
                    dv_label.write(str(label_dict[label]) + '\n')
                else:
                    tr_file.write(sent + '\n')
                    tr_label.write(str(label_dict[label]) + '\n')

    with codecs.open(test_file, 'rb', encoding='latin-1') as datafile, \
            codecs.open(os.path.join(test_dir, 'data.txt'), 'wb', encoding='latin-1') as te_file, \
            codecs.open(os.path.join(test_dir, 'label.txt'), 'wb', encoding='latin-1') as te_label:
            for line in datafile:
                es = line.strip().split()
                label, fine_label = es[0].split(':')
                sent = ' '.join(es[1:])
                te_file.write(sent + '\n')
                te_label.write(str(label_dict[label]) + '\n')
    return train_dir, dev_dir, test_dir

if __name__ == '__main__':
    print('=' * 80)
    print('Pre-processing TREC dataset')
    print('=' * 80)

    base_dir = os.path.dirname('.')
    data_dir = os.path.join(base_dir, 'data')
    trec_dir = os.path.join(data_dir, 'trec')

    # split into separate files
    tr_dir, dv_dir, te_dir = split(trec_dir)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(trec_dir, '*/data.txt')),
        os.path.join(trec_dir, 'wvocab.txt'),
        os.path.join(trec_dir, 'cvocab.txt'))
    build_vocab(
        glob.glob(os.path.join(trec_dir, '*/data.txt')),
        os.path.join(trec_dir, 'wvocab-cased.txt'),
        os.path.join(trec_dir, 'cvocab-cased.txt'),
        lowercase=False)
