# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:21:51 2016

@author: mikol
"""
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal.pool import pool_2d
from theano.ifelse import ifelse

class Layer(object):
    def __init__(self, rng, input, input_sh, name=''):
        self.rng = rng
        self.input = input
        self.input_sh = input_sh
        self.output_sh = (0,)
        self.name = name
        self.params = []
        self.speeds = []
    
    def __repr__(self):
        return (self.name.ljust(25) + ' ' + repr(self.input_sh) + 
                ' --> ' + repr(self.output_sh))
        

class ConvLayer(Layer):
    def __init__(self, rng, input, input_sh, filter_sh, 
                 activation=T.nnet.sigmoid, name='ConvLayer'):
#         print('Conv input shape: ' + repr(input.size))
        assert input_sh[1] == filter_sh[1]
        super().__init__(rng, input, input_sh, name)
        self.output_sh = (
            input_sh[0], 
            filter_sh[0], 
            input_sh[2] - filter_sh[2] + 1, 
            input_sh[3] - filter_sh[3] + 1
            )
        # weights
        w_bound = np.sqrt(filter_sh[1]*filter_sh[2]*filter_sh[3])
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-1/w_bound, high=1/w_bound, size=filter_sh),
                dtype=theano.config.floatX), 
            name='W' + name)
        # bias
        b_sh = (filter_sh[0], )
#         print(b_sh)
        b_bound= .5
        self.b = theano.shared(
            np.asarray(
                rng.uniform(low=-b_bound, high=b_bound, size=b_sh),
                dtype=theano.config.floatX),
            name='b' + name)
#         self.b.tag.test_value = np.array([0])
#         b_printed = theano.printing.Print('self.b = ')(self.b)
#         bshape_printed = theano.printing.Print('b_shape = ')(self.b.dimshuffle('x', 0, 'x', 'x').shape)
        # param list
        self.params = [self.W, self.b]
        self.speeds = [np.sqrt(np.prod(input_sh[1:]))] * 2
        # output
        conv_out = conv2d(input, self.W, filter_shape=filter_sh, input_shape=input_sh)
        lin_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = lin_out if (activation is None) else activation(lin_out)

class MaxPoolLayer(Layer):
    def __init__(self, rng, input, input_sh, pool_sh, 
                 activation=T.nnet.sigmoid, name='MaxPoolLayer'):
        super().__init__(rng, input, input_sh, name)
        self.output_sh = (
            input_sh[0],
            input_sh[1],
            input_sh[2] // pool_sh[0],
            input_sh[3] // pool_sh[1]
            )
        # output
        lin_out = pool_2d(input=input, ds=pool_sh, ignore_border=True)
        self.output = lin_out if (activation is None) else activation(lin_out)

class AddMaxLayer(Layer):
    def __init__(self, rng, input, input_sh, name='AddMaxLayer'):
        super().__init__(rng, input, input_sh, name)
        self.output_sh = (input_sh[0], input_sh[1] + 1)
        self.output = T.concatenate([self.input, 
                                     T.max(self.input, axis=1, keepdims=True)], 
                                    axis=1)

class FullyConnectedLayer(Layer):
    def __init__(self, rng, input, input_sh, n_out, 
                 W_0=None, b_0=None, activation=T.nnet.sigmoid, 
                 name='FullyConnectedLayer', p=.5, fit_intercept=True):
        super().__init__(rng, input, input_sh, name)
#        print('FullyConnected input shape: ' + repr(input.size))
        self.output_sh = (input_sh[0], n_out)
        self.W_sh = (input_sh[1], n_out)
        self.n_out = n_out
        self.activation = activation
        self.fit_intercept = int(fit_intercept)
        # dropout
        self.default_p = p
        self.p = theano.shared(p)
#        self.p.set_value(p
        self.drop_input = theano.shared(self._dropout(),
                                        name='drop_input' + name, 
                                        borrow=True)
        # weights
        if W_0 is None:
            W_0 = self._default_W()
        self.W = theano.shared(W_0, name='W' + name, borrow=True)
        # bias
        if b_0 is None:
            b_0 = self._default_b()
        self.b = theano.shared(b_0, name='b' + name, borrow=True)
        # param list
        self.params = [self.W] + self.fit_intercept * [self.b]
        self.speeds = [np.sqrt(input_sh[1])] * 2
        # output
        input_ = ifelse(T.gt(self.p.eval(), 0), 
                        input * self.drop_input / (1 - self.p),
                        input)
        lin_out = ifelse(
            self.fit_intercept, 
            T.dot(input_, self.W) + self.b.repeat(repeats=input_sh[0], axis=0), 
            T.dot(input_, self.W)
        )
        self.output = lin_out if (activation is None) else activation(lin_out)
#        self.L1 = abs(self.W).sum() + abs(self.b).sum()
#        self.L2 = (self.W ** 2).sum() + (self.b ** 2).sum()
    
    def _default_W(self, mult=1):
        bound = np.sqrt(6/(self.W_sh[0] + self.W_sh[1])) * mult
        if self.activation == T.nnet.sigmoid:
            bound *= 4
        W_0 = np.asarray(
            self.rng.uniform(
                low=-bound, 
                high=bound, 
                size=self.W_sh
            ),
            dtype=theano.config.floatX
        )
        return W_0
    
    def _dropout(self):
        return np.asarray(
            np.repeat(
                self.rng.binomial(1, 1 - self.p.eval(), (1, self.input_sh[1])),
                self.input_sh[0],
                axis=0
                ),
            dtype=theano.config.floatX
            )
    
    def dropout_on(self, p=None):
        self.p.set_value(self.default_p if (p is None) else p)
        self.drop_input.set_value(self._dropout())
        
    def dropout_off(self):
        self.p.set_value(0)
    
    def _default_b(self):
        if (self.activation == T.nnet.relu) and self.fit_intercept:
            return (np.ones((1, self.n_out), dtype=theano.config.floatX) * .0)
        return np.zeros((1, self.n_out), dtype=theano.config.floatX)
        
    def reset_params(self, mult=1):
        self.W.set_value(self._default_W(mult))
        self.b.set_value(self._default_b())
        
    def pretrain(self, X, batch_size, n_batches, rate=.01, rate_min=.00001, 
                 steps=10000, use_b=True):
#        if self.mirror is None:
#            mirror = FullyConnectedLayer(self.rng, self.output, 
#                                         self.output_sh, self.input_sh[1], 
#                                         activation=self.activation)
        self.dropout_off()
        idx = T.lscalar(name='idx')
        sh_rate = theano.shared(np.asarray([rate], dtype=theano.config.floatX))
        Loss = ((self.input - T.dot(self.output, self.W.transpose()))**2).mean()
        pars = self.params[: 1 + use_b]# + mirror.params[: 1 + use_b]
        Cost = Loss + .005 * T.sum([(p**2).sum() for p in pars])        
        _grads = [g.clip(-10,10) for g in T.grad(Cost, pars)]
        norm = T.sqrt(T.sum([T.sum(g**2) for g in _grads], keepdims=True))
        minnorm, maxnorm = .1, 10
        norm1 = T.clip(norm, maxnorm, np.inf)/maxnorm * T.clip(norm, 1e-10, minnorm)/minnorm
        grads = [g/norm1 for g in _grads]
        gradf = theano.function([self.input], grads)
        gradif = theano.function([idx], grads, givens={self.input: X[idx * batch_size: (idx + 1) * batch_size, :]})
        updates = [(p, p - sh_rate[0] * g) for p, g in zip(pars, grads)]
        pretrain_model = theano.function(
            [idx],
            Cost,
            updates=updates,
            givens={self.input: X[idx * batch_size: (idx + 1) * batch_size, :]}
        )
        loss_model = theano.function(
            [idx], Loss,
            givens={self.input: X[idx * batch_size: (idx + 1) * batch_size, :]}
        )
        print('Pretraining ' + self.name)
        _R2 = - np.inf
        for i in range(steps):
            pretrain_model(i % n_batches)
            sh_rate.set_value([np.min([.05, 1/(1/rate + i/steps * (1/rate_min - 1/rate))])])
            if i%(min(steps // 10, 25000)) == 0:
                self.W.set_value(self.W.eval() / (1 - self.p.eval()))
                loss = np.mean([loss_model(i) for i in np.arange(n_batches)])
                R2 = 1 - loss/(X.eval().std()**2)
                self.W.set_value(self.W.eval() * (1 - self.p.eval()))
                print('iter %d, learning_rate %.4f, loss %.4f, R2 %.2f' % (i, sh_rate.eval()[0], loss, R2))
                if (R2 > .99):# or (R2*.999 < _R2):
                    break
                _R2 = R2
#                print((i, self.W.eval()))
#                print((i, np.dot(mirror.W.eval(), self.W.eval())))
#                print((i, mirror.b.eval(), self.b.eval()))
                print((i, '|grad|^2: ', np.sum([np.sum(g**2) for g in gradif(i % n_batches)])))
#            if i%(steps // 1000) == 0:
#                print(self.W.eval())
        self.dropout_on()
        
    def get_pretrain_sample(self, tv_input, tv_idx, shared_input, batch_size,
                            batches):
        givens = {tv_input: shared_input[tv_idx * batch_size: (tv_idx + 1) * batch_size]}
        forward = theano.function([tv_idx],
                                  self.input,
                                  givens=givens)
        pretrained = np.asarray(np.concatenate([forward(i) for i in np.arange(batches)]),
                                dtype=theano.config.floatX)
        print(pretrained.shape)
        return theano.shared(pretrained, borrow=True)
        
        
class SeparatedFCLayer(Layer):
    def __init__(self, rng, input, input_sh, n_out, common_params=True, 
                 W_0=None, b_0=None, activation=T.nnet.sigmoid, 
                 name='SeparatedFCLayer', p=.5):
        super().__init__(rng, input, input_sh, name)
        self.output_sh = (input_sh[0], n_out, input_sh[2])
        self.sublayers = [FullyConnectedLayer(
            rng, input[:, :, 0], (input_sh[0], input_sh[1]), n_out,
            activation=activation)]
        self.common_params = common_params
        self.params = self.sublayers[0].params
        self.default_p = p
        for k in np.arange(1, input_sh[2]):
            self.sublayers.append(FullyConnectedLayer(
                rng, input[:, :, k], (input_sh[0], input_sh[1]), n_out, 
                activation=activation))
            if common_params:
                self.sublayers[-1].W = self.sublayers[0].W
                self.sublayers[-1].b = self.sublayers[0].b
            else:
                self.params += self.sublayers[k].params
        self.output = T.stack([sl.output for sl in self.sublayers], axis=2)
    
    def reset_params(self, mult=1):
        for sl in self.sublayers:
            sl.reset_params(mult)
            if self.common_params:
                break
    
    def dropout_on(self, p=None):
        if p is None:
            p = self.default_p
        for sl in self.sublayers:
            sl.dropout_on(p)
    
    def dropout_off(self):
        for sl in self.sublayers:
            sl.dropout_off()
        

class ScaleLayer(Layer): #Assumes positive input
    def __init__(self, rng, input, input_sh, exp=False, scale=1/(24*3600), 
                 name='ScaleLayer'):
        super().__init__(rng, input, input_sh, name)
        if scale is not None:
            self.l = theano.shared(scale, name='lambda' + name)
            self.params = [self.l]
            input0 = T.nnet.softplus(input * self.l)
        else:
            input0 = T.nnet.softplus(input)
        input_ = T.exp(-input0) if exp else input0
        div = input_.sum()
        self.output_sh = input_sh
        self.output = input_/(div + .0000001*(div == 0))
        
        
class JoinLayer(Layer):
    def __init__(self, rng, input1, input2, input_sh, name='JoinLayer'):
        super().__init__(rng, input, input_sh, name)
        self.output_sh = input_sh
        self.output = input1 * input2
 
    def __repr__(self):
        return (self.name + ' ' + repr([self.input_sh] * 2) + ' --> ' + repr(self.output_sh))       
        
#class SoftMaxLayer(Layer):
#    def __init__(self, rng, input, input_sh, output_sh, name='SoftMaxLayer'):
#        super().__init__(rng, input, input_sh, name)
#        self.output_sh=output_sh
#        self.


        
#QBch = pd.read_pickle('quoteBook_changes.pkl')
#lags = 10
#features = QBch.shape[1]
#in0 = np.array(QBch[:lags]).reshape(1, 1, lags, features)
#in0.shape