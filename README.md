# MDSENT:  A Supervised System for Text Classification

An implementation of the MDSENT architectures described in the paper [MDSENT at SemEval-2016 Task 4: A Supervised System for Message
Polarity Classification](http://www.aclweb.org/anthology/S/S16/S16-1020.pdf) by Hang Gao and Tim Oates.

## Requirements
* [Theano](http://deeplearning.net/software/theano/)
* [Numpy](http://www.numpy.org/)
* [Lasagne](https://github.com/Lasagne/Lasagne)
* Python 2.7

## Usage
First run the following script
~~~~
./fetch_and_preprocess.sh
~~~~

This downloads the following data:
* [TREC dataset](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (question classification task)
* [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Clawl 840B) -- **PS:** the download takes around 2GB

Alternatively, the download and preprocessing scripts can be called individually.

## Training
This implementation only supports classification task. To train a network, run,
~~~
python2.7 train.py --train data/trec/train --dev data/trec/dev --batch 15 --epoch 10 --save model.pkg --word_vectors data/glove/glove.840B.300d.txt --word_vocab data/trec/wvocab-cased.txt --char_vocab data/trec/cvocab-cased.txt --w_filter 3 100 --w_filter 4 100 --w_filter 5 100 --c_filter 3 100 --c_filter 4 100 --c_filter 5 100 --num_class 6
~~~
where:
  * `train`: the location of preprocessed training data
  * `dev`: the location of preprocessed development data
  * `batch`: the batch size
  * `epoch`: number of training epochs
  * `save`: the location to save trained model
  * `word_vectors`: the location of pre-trained word embeddings
  * `char_vectors`: the location of pre-trained character embeddings
  * `word_vocab`: the location of preprocessed vocabulary for words
  * `char_vocab`: the location of preprocessed vocabulary for characters
  * `w_filter`: specifies a type of convolutional filter for word-based input with its height(int) and number(int)
  * `c_filter`: specifies a type of convolutional filter for character-based input with its height(int) and number(int)
  * `num_class`: number of classes for the classification task

## Testing
To make predictions with a trained model, run,
~~~
python2.7 test.py --test data/trec/test --model model.pkg --word_vocab data/trec/wvocab-cased.txt --char_vocab data/trec/cvocab-cased.txt
~~~
where:
  * `test`: the location of preprocessed test data
  * `model`: the location of trained model
  * `word_vocab`: the location of preprocessed vocabulary for words
  * `char_vocab`: the location of preprocessed vocabulary for characters

PS: the implementation is in Theano, thus it is recommended that floatX is set to float32 in theano flags to avoid possible precision problems.
