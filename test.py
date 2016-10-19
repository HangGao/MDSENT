import argparse
import sys
from syslib import read_vocab, read_trec, read_embeddings
import numpy as np
from mdsent import Model


parser = argparse.ArgumentParser(description='A Supervised System for Text Classification')
parser.add_argument('-t', '--test', action='store', required=True, help='location of preprocessed test data')
parser.add_argument('-m', '--model', action='store', required=True, help='location of stored model')
parser.add_argument('-p', '--word_vocab', action='store', required=True, help='location of word vocabulary')
parser.add_argument('-P', '--char_vocab', action='store', required=True, help='location of char vocabulary')

args = parser.parse_args(sys.argv[1:])
print args

wvocab, cvocab = read_vocab(args.word_vocab, args.char_vocab)

test_data = read_trec(args.test, wvocab, cvocab)
test_w_x, test_c_x, test_y = test_data['np_wsents'], test_data['np_csents'], test_data['np_labels']

model = Model.load(args.model)
model.test((test_w_x, test_c_x, test_y))
