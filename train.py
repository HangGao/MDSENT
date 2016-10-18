import argparse
import sys
from syslib import read_vocab, read_trec, read_embeddings
import numpy as np
from mdsent import Model


parser = argparse.ArgumentParser(description='A Supervised System for Text Classification')
parser.add_argument('-t', '--train', action='store', required=True, help='location of preprocessed training data')
parser.add_argument('-d', '--dev', action='store', required=True, help='location of preprocessed dev data')
parser.add_argument('-b', '--batch', action='store', default=10, type=int, help='batch size. Default:10')
parser.add_argument('-e', '--epoch', action='store', default=10, type=int, help='number of epochs')
parser.add_argument('-s', '--save', action='store', help='store trained model to the location')
parser.add_argument('-v', '--word_vectors', action='store', help='location of pretrained word embeddings')
parser.add_argument('-V', '--char_vectors', action='store', help='location of pretrained char embeddings')
parser.add_argument('-w', '--w_filter', action='append', nargs=2,
                    help='specify a type of word based filter with its height and number')
parser.add_argument('-c', '--c_filter', action='append', nargs=2,
                    help='specify a type of character based filter with its height and number')
parser.add_argument('-n', '--num_class', action='store', type=int, default=6, help='number of classes')

args = parser.parse_args(sys.argv[1:])
print args

pre_trained_w_embs = None
pre_trained_c_embs = None

wvocab, cvocab = read_vocab('data/trec/wvocab-cased.txt', 'data/trec/cvocab-cased.txt')
if args.word_vectors is not None:
    pre_trained_w_embs = read_embeddings(wvocab, args.word_vectors)
else:
    pre_trained_w_embs = np.random.uniform(-0.25, 0.25, (len(wvocab)+1, 300))
    pre_trained_w_embs[0, :] = 0.

if args.char_vectors is not None:
    pre_trained_c_embs = read_embeddings(cvocab, args.char_vectors)
else:
    pre_trained_c_embs = np.random.uniform(-0.25, 0.25, (len(cvocab)+1, 300))
    pre_trained_c_embs[0, :] = 0.

w_grams = [int(param[0]) for param in args.w_filter]
w_nfs = [int(param[1]) for param in args.w_filter]

c_grams = [int(param[0]) for param in args.c_filter]
c_nfs = [int(param[1]) for param in args.c_filter]

mlp_layers = (args.num_class,)

train_data = read_trec('data/trec/train', wvocab, cvocab)
dev_data = read_trec('data/trec/dev', wvocab, cvocab)
train_w_x, train_c_x, train_y = train_data['np_wsents'], train_data['np_csents'], train_data['np_labels']
dev_w_x, dev_c_x, dev_y = dev_data['np_wsents'], dev_data['np_csents'], dev_data['np_labels']

model = Model(pre_trained_w_embs=pre_trained_w_embs,
              pre_trained_c_embs=pre_trained_c_embs,
              w_grams=tuple(w_grams),
              w_nfs=tuple(w_nfs),
              c_grams=tuple(c_grams),
              c_nfs=tuple(c_nfs),
              mlp_layers=mlp_layers,
              )
model.train((train_w_x,train_c_x, train_y),
            (dev_w_x, dev_c_x, dev_y), n_epochs=args.epoch)

if args.save is not None:
    print '... saving model'
    model.save(args.save)