import lasagne
from lasagne.nonlinearities import softmax
from lasagne.init import Uniform
from lasagne.layers import InputLayer, EmbeddingLayer, ReshapeLayer, Conv2DLayer, Layer, ConcatLayer, DropoutLayer, \
    DenseLayer, get_output, get_all_params, get_all_param_values, set_all_param_values
import theano.tensor as T
import theano
import cPickle as pickle
import time
import numpy as np
import sys


def negative_log_likelihood(prediction, target):
    return -T.mean(T.log(prediction)[T.arange(target.shape[0]), target])


def std_printf(msg):
    sys.stdout.write('\r' + str(msg))
    sys.stdout.flush()


class MaxLayer(Layer):

    def __init__(self, incoming, axis=1, **kwargs):
        super(MaxLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_for(self, input, **kwargs):
        return T.max(input, axis=self.axis)


class Model(object):
    def __init__(self,
                 pre_trained_w_embs=None,
                 pre_trained_c_embs=None,
                 w_grams=(3, 4, 5),
                 w_nfs=(50, 50, 50),
                 c_grams=(4, 5, 6),
                 c_nfs=(50, 50, 50),
                 mlp_layers=(2,),
                 mlp_dropouts=(0.5,),
                 mlp_nonlinearities=(softmax,),
                 opt_method=lasagne.updates.adadelta,
                 opt_args={'learning_rate': 0.1, 'rho': 0.95, 'epsilon': 1e-6},
                 **kwargs
                 ):
        parameters = locals()
        parameters.update(kwargs)
        self.parameters = parameters

        assert pre_trained_w_embs is not None
        assert pre_trained_c_embs is not None
        if isinstance(pre_trained_w_embs, tuple):
            w_vocab_size = pre_trained_w_embs[0]
            w_emb_dim = pre_trained_w_embs[1]
            pre_trained_w_embs = Uniform(0.25).sample((w_vocab_size, w_emb_dim))
        else:
            w_vocab_size = pre_trained_w_embs.shape[0]
            w_emb_dim = pre_trained_w_embs.shape[1]
        t_pre_trained_w_embs = theano.shared(pre_trained_w_embs, name='w_embs', borrow=True)

        if isinstance(pre_trained_c_embs, tuple):
            c_vocab_size = pre_trained_c_embs[0]
            c_emb_dim = pre_trained_c_embs[1]
            pre_trained_c_embs = Uniform(0.25).sample((c_vocab_size, c_emb_dim))
        else:
            c_vocab_size = pre_trained_c_embs.shape[0]
            c_emb_dim = pre_trained_c_embs.shape[1]
        t_pre_trained_c_embs = theano.shared(pre_trained_c_embs, name='c_embs', borrow=True)

        w_sents = T.imatrix(name='w_sents')
        c_sents = T.imatrix(name='c_sents')
        labels = T.ivector(name='labels')

        w_input = InputLayer((None, None), input_var=w_sents)
        c_input = InputLayer((None, None), input_var=c_sents)

        w_embs = EmbeddingLayer(w_input, input_size=w_vocab_size, output_size=w_emb_dim, W=t_pre_trained_w_embs)
        w_embs = ReshapeLayer(w_embs, ([0], 1, [1], [2]))
        c_embs = EmbeddingLayer(c_input, input_size=c_vocab_size, output_size=c_emb_dim, W=t_pre_trained_c_embs)
        c_embs = ReshapeLayer(c_embs, ([0], 1, [1], [2]))

        conv_layers = []
        for w_gram, w_nf in zip(w_grams, w_nfs):
            conv_layer = Conv2DLayer(w_embs, w_nf, (w_gram, w_emb_dim), pad=(w_gram - 1, 0))
            # shape (batch_size, nfs[i], 1, 1)
            pooled_layer = MaxLayer(conv_layer, axis=2)
            # shape (batch_size, nfs[i])
            flatten_layer = ReshapeLayer(pooled_layer, ([0], [1]))
            conv_layers.append(flatten_layer)

        for c_gram, c_nf in zip(c_grams, c_nfs):
            # shape (batch_size, nfs[i], num_features, 1)
            conv_layer = Conv2DLayer(c_embs, c_nf, (c_gram, c_emb_dim), pad=(c_gram - 1, 0))
            # shape (batch_size, nfs[i], 1, 1)
            pooled_layer = MaxLayer(conv_layer, axis=2)
            # shape (batch_size, nfs[i])
            flatten_layer = ReshapeLayer(pooled_layer, ([0], [1]))
            conv_layers.append(flatten_layer)

        network = ConcatLayer(conv_layers, axis=1)

        for mlp_layer, mlp_dropout, mlp_nonlinearity in zip(mlp_layers, mlp_dropouts, mlp_nonlinearities):
            if mlp_dropout is not None:
                network = DropoutLayer(network, p=mlp_dropout)
            network = DenseLayer(network, num_units=mlp_layer, nonlinearity=mlp_nonlinearity)

        self.network = network

        # Create a loss expression for training, i.e., negative log likelihood we want to maximize):
        train_predict = get_output(network)
        train_loss = negative_log_likelihood(train_predict, labels)
        train_acc = T.sum(T.eq(T.argmax(train_predict, axis=1), labels), axis=0)

        # We could add some weight decay as well here, see lasagne.regularization.

        # Here to create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = get_all_params(network, trainable=True)
        updates = opt_method(train_loss, params, **opt_args)
        # correct updates for embeddings[0] by resetting it to its initial value
        updates[t_pre_trained_w_embs] = T.set_subtensor(updates[t_pre_trained_w_embs][0, :], t_pre_trained_w_embs[0])
        updates[t_pre_trained_c_embs] = T.set_subtensor(updates[t_pre_trained_c_embs][0, :], t_pre_trained_c_embs[0])

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = get_output(network, deterministic=True)
        test_loss = negative_log_likelihood(test_prediction, labels)
        test_acc = T.sum(T.eq(T.argmax(test_prediction, axis=1), labels), axis=0)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        self.train_fn = theano.function([w_sents, c_sents, labels], [train_loss, train_acc], updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        self.test_fn = theano.function([w_sents, c_sents, labels], [test_loss, test_acc])

    def get_params(self):
        return get_all_param_values(self.network)

    def set_params(self, param_values):
        set_all_param_values(self.network, param_values)

    def save(self, file_path):
        pickle.dump((self.parameters, self.get_params()), open(file_path, 'wb'))

    @staticmethod
    def load(file_path):
        parameters, param_vals = pickle.load(open(file_path, 'rb'))
        network = Model(**parameters)
        network.set_params(param_vals)
        return network

    def train(self,
              train_set,
              dev_set,
              batch_size=10,
              n_epochs=10,
              ):

        # start training over mini-batches
        print '... training'
        epoch = 0
        best_dev_acc = 0
        best_params = None
        while epoch < n_epochs:
            start_time = time.time()
            epoch += 1
            batch = 0

            train_set_w_x, train_set_c_x, train_set_y = train_set
            dev_set_w_x, dev_set_c_x, dev_set_y = dev_set

            data_size = train_set_w_x.shape[0]
            indices = np.random.permutation(data_size)

            # train the network
            for i in xrange(0, data_size, batch_size):
                batch += 1
                real_batch_size = min(i + batch_size, data_size) - i
                batch_train_set_w_x = train_set_w_x[indices[i: i + real_batch_size]]
                batch_train_set_c_x = train_set_c_x[indices[i: i + real_batch_size]]
                batch_train_set_y = train_set_y[indices[i: i + real_batch_size]]

                costs = self.train_fn(batch_train_set_w_x, batch_train_set_c_x, batch_train_set_y)
                std_printf('epoch %d  batch %d  train_loss: %.3f  train_acc: %.2f %%' %
                       (epoch, batch, costs[0], costs[1] * 100 / real_batch_size))

            overall_acc = 0.
            num_inst = 0
            # eval the network on train data
            for i in xrange(0, data_size, batch_size):
                real_batch_size = min(i + batch_size, data_size) - i
                batch_train_set_w_x = train_set_w_x[indices[i: i + real_batch_size]]
                batch_train_set_c_x = train_set_c_x[indices[i: i + real_batch_size]]
                batch_train_set_y = train_set_y[indices[i: i + real_batch_size]]

                num_inst += real_batch_size
                costs = self.test_fn(batch_train_set_w_x, batch_train_set_c_x, batch_train_set_y)
                overall_acc += costs[1]

            train_acc = overall_acc / data_size
            std_printf('epoch %d   train prediction acc: %.2f %%' % (epoch, train_acc * 100))

            overall_acc = 0.
            dev_size = dev_set_w_x.shape[0]
            for i in xrange(0, dev_size, batch_size):
                real_batch_size = min(i + batch_size, dev_size) - i
                batch_dev_set_w_x = dev_set_w_x[i: i + real_batch_size]
                batch_dev_set_c_x = dev_set_c_x[i: i + real_batch_size]
                batch_dev_set_y = dev_set_y[i: i + real_batch_size]

                costs = self.test_fn(batch_dev_set_w_x, batch_dev_set_c_x, batch_dev_set_y)
                overall_acc += costs[1]

            dev_acc = overall_acc / dev_size
            std_printf('epoch %d   dev prediction acc: %.2f %%' % (epoch, dev_acc * 100))

            std_printf('epoch: %i, training time: %.2f secs, train acc: %.2f %%, dev acc: %.2f %%\n' %
                   (epoch, time.time() - start_time, train_acc * 100., dev_acc * 100.))
            if dev_acc >= best_dev_acc:
                best_dev_acc = dev_acc
                best_params = self.get_params()

        self.set_params(best_params)
        return best_dev_acc

    def test(self,
             test_set,
             batch_size=10,
             ):
        print '... testing'
        test_set_w_x, test_set_c_x, test_set_y = test_set
        test_size = test_set_w_x.shape[0]

        overall_acc = 0.
        # eval the network on train data
        for i in xrange(0, test_size, batch_size):
            real_batch_size = min(i + batch_size, test_size) - i
            batch_test_set_w_x = test_set_w_x[i: i + real_batch_size]
            batch_test_set_c_x = test_set_c_x[i: i + real_batch_size]
            batch_test_set_y = test_set_y[i: i + real_batch_size]
            costs = self.test_fn(batch_test_set_w_x, batch_test_set_c_x, batch_test_set_y)
            overall_acc += costs[1]
        test_acc = overall_acc / test_size
        std_printf('test prediction acc: %.2f %%' % (test_acc * 100))

