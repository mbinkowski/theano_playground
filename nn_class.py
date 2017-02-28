# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 12:48:14 2016

@author: mbinkowski
"""

import pandas as pd
import imp
import time
import quotebook_utils
imp.reload(quotebook_utils)
from quotebook_utils import *
import nn_layers
imp.reload(nn_layers)
from nn_layers import *
import nn_utils
imp.reload(nn_utils)
from nn_utils import *
import numpy as np

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
import pickle
import datetime as dt

class NN(object):
    def __init__(self, data, data_div=[.8, .1], 
                 batch_size=16, max_norm=1000,
                 act=T.nnet.relu):
#        self.layer_size = layer_size
        self.batch_size = batch_size
        self.max_norm = max_norm
#        self.dropout_rate = dropout_rate
#        self.dropout_rate_1 = dropout_rate_1
        self.default_act = act
        self.layers = []
        self.params = []
        self.trainer = None
        self.Loss = None
        self.Cost = None
        self.grads = None
#        self.print_grad = True
#        self.iters = [0]
#        self.improvement = [1] * 5
        self.rng = np.random.RandomState(21312)
        self._set_data(data, data_div)
        try:
            var = ((data['y'][start['test']:] - data['y'][start['test']:].mean())**2).mean()
            print('Zero prediction R2 = %.2f' % (1 - (data['y'][start['test']:]**2).mean()/var))
            print('Mean prediction R2 = %.2f' % (1 - ((data['y'][start['test']:] - data['y'][:n_train].mean())**2).mean()/var))
            print('Last return prediction R2 = %.2f' % 
                  (1 - ((data['y'][start['test']:] - data['X'][start['test']:, :, 0, dir, 0])**2).mean()/var))
        except Exception as e:
            pass
        
        self.tv = {
            'y': T.matrix(name='y'),
            'mask': T.matrix(name='mask'),
            'idx': T.lscalar(name='idx')
        }
        Xsh = data['X'].shape
        if len(Xsh) == 2:
            self.tv['X'] = T.matrix(name='X')
        elif len(Xsh) == 3:
            self.tv['X'] = T.tensor3(name='X')
        elif len(Xsh) == 4:
            self.tv['X'] = T.tensor4(name='X')
        else:
            raise Exception("Wrong dimension: data['X'].shape = " + repr(Xsh))
        self.time0 = time.time()    
      
    def add_layer(self, layer):
        layer.name = '(' + repr(len(self.layers)) + '): ' + layer.name 
        self.layers.append(layer)
        self.params += layer.params
        print(repr(layer))
        
    def add_layers(self, layers):
        for l in layers:
            self.add_layer(l)
        
    def pretrain_layers(self, rate=.05, rate_min=.01, steps=250000, I=None):
        if I is None:
            to_pretrain = self.layers[1:]
            I = np.arange(1, len(self.layers))
            layer = self.layers[0]
            if hasattr(layer, 'pretrain'):
                layer.pretrain(self.shared['X'], self.batch_size, 
                               self.batches['train'], rate=rate, 
                               rate_min=rate_min, steps=steps, use_b=True)
            else:
                print("Layer 0 '" + layer.name + "' is not pretrainable")
        else:
            to_pretrain = [self.layers[i] for i in I]
        for i, layer in zip(I, to_pretrain):
            if hasattr(layer, 'pretrain'):
                pretrained = layer.get_pretrain_sample(
                    self.tv['X'], self.tv['idx'], 
                    self.shared['X'], self.batch_size, 
                    self.batches['train']
                    )
                layer.pretrain(pretrained, self.batch_size, 
                               self.batches['train'], rate=rate, 
                               rate_min=rate_min, steps=steps, use_b=True)
            else:
                print("Layer %d '%s' is not pretrainable" % (i, layer.name))
                
    def get_param_values(self):
        return [p.eval() for p in self.params]
        
    def set_loss(self, loss, _type='classifier', forward_layer=-1):
        self.Loss = loss[0]
        self.Cost = loss[1]
        self.grads = [g for g in T.grad(self.Cost, self.params)]
        
        ### Defining theano models
        self.tf = {
            'valid': self._ev_model(self.Loss, 'valid'),
            'test': self._ev_model(self.Loss, 'test'),
            'train': self._ev_model(self.Cost, 'train'),
            'train_loss': self._ev_model(self.Loss, 'train_loss', 'train'),
            'grad': self._ev_model(self.grads, 'grad', 'train'),
            'forward': self._ev_model(self.layers[forward_layer].output, 
                                      'forward', 'all')
            }
        print('models ready')
#        self.best_losses = {
#            'last_valid': np.inf,
#            'valid': np.inf, 
#            'test': np.inf, 
#            'test R2': -np.inf, 
#            'params': []
#            }
#        self.loss_history = {
#            'train': [],
#            'valid': [],
#            'learning_rate': []
#        }
        self.cum_sizes = np.cumsum([0] + [np.prod(p.get_value().shape) for p in self.params])
        self.type = _type
        if _type == 'classifier':
            self.type_loss_stats = classifier_loss_stats 
        else: 
            self.type_loss_stats = regressor_loss_stats
    
    def _ev_model(self, loss, name, range_name=None):
        if range_name is None:
            range_name = name
        return ev_model(self.tv, self.shared, self.data, self.batch_size, loss,
                        self.start[range_name], self.batches[range_name], name)
                        
    def _set_data(self, data, data_div):
        Xsh0 = data['X'].shape[0]
        self.batches = {
            'train': int((Xsh0 * data_div[0]) // self.batch_size),
            'valid': int((Xsh0 * data_div[1]) // self.batch_size),
            'all': int(Xsh0 // self.batch_size)
            }
        self.batches['test'] = self.batches['all'] - self.batches['train'] - self.batches['valid']
        self.start = {
            'train': 0,
            'valid': self.batches['train'] * self.batch_size,
            'test': (self.batches['train'] + self.batches['valid']) * self.batch_size,
            'all': 0
            }
        order = np.r_[
            np.random.permutation(self.batches['train'] * self.batch_size), 
            np.arange(self.batches['train'] * self.batch_size, Xsh0)
            ]
        self.data = dict([(k, v[order]) for k, v in data.items()])
        self.shared = dict(
            [(k, theano.shared(np.asarray(v, dtype=theano.config.floatX), 
                               borrow=True)) for k, v in self.data.items()]
            )
   
    def _set_params(self, flat_param_values):
        if any(np.isnan(flat_param_values).ravel()):
            print('nans passed to set_params')
        scaled = False
        new_params, scaled_params = [], []
        scaled_flat_param_values = copy.deepcopy(flat_param_values)
        for i, p in enumerate(self.params):
            sh = p.get_value().shape
            l, r = int(self.cum_sizes[i]), int(self.cum_sizes[i+1])
            v = flat_param_values[l: r].reshape(sh)
            new_params.append(copy.deepcopy(v))
    #         if len(sh) == 2:
    #             for j in range(sh[1]):
    #                 norm_ratio = max_norm/(v[:, j]**2).sum()
    #                 if norm_ratio < 1:
    #                     v[:, j] *= np.sqrt(norm_ratio)
            if (np.abs(v) > 1e5).any():
                print('big v')
                v = v.clip(-1e5, 1e5)
                scaled = True
            div = (v**2).sum(axis=0, keepdims=True)
            if np.isnan(div).any():
                print('dvi nan')
                print(v)
    #             print('0 div ' + repr(np.sum(div==0)))
    #             temp = max_norm/div
    #             print('nann'  + repr(np.isnan(temp).any()))
            div2 = (np.sqrt(div/self.max_norm)).clip(1, np.inf)
            if (div2 > 1).any():
                scaled = True
                v /= div2
                scaled_flat_param_values[l: r] = v.flatten()
            if (div2==0).any():
                print('div2 0 val')
            if np.isnan(div2).any():
                print('dvi2 nan')
            scaled_params.append(v)
        if scaled:
            flat_grad_values = -self.trainer.grad_function(flat_param_values)
            if not any(np.isnan(flat_grad_values)):
                for p, v in zip(self.params, scaled_params):
                    p.set_value(v)
    #             print('rescaling')
                return scaled_flat_param_values, flat_grad_values 
        for p, v in zip(self.params, new_params):
            p.set_value(v)
        return flat_param_values, None    
                          
    def set_param_values(self, pars):
        for p, v in zip(self.params, pars):
            if p.eval().shape == v.shape:
                p.set_value(v)
            else:
                mess = 'NN.set_param_values error: Dimension mismatch. \n'
                mess += ('len(self.params) = %d, len(values) = %d \n' % (len(self.params), len(pars)))
                mess += ('p.shape = ' + repr(p.eval().shape) + ', v.shape = ' + repr(v.shape))
                raise Exception(mess)

class FFN(NN):
    def __init__(self, data, data_div=(.8, .1), 
                 layer_no=3, layer_size=128, batch_size=16, max_norm=10,
                 dropout_rate=0, dropout_rate_1=0, act=T.nnet.relu):
        super().__init__(data, data_div=data_div, batch_size=batch_size, 
                         max_norm=max_norm, act=act)
        self.layer_size = layer_size
        self.dropout_rate = dropout_rate
        self.dropout_rate_1 = dropout_rate_1
        for i in np.arange(layer_no):
            self.add_layer(final=(i + 1 == layer_no))
        
    def add_layer(self, layer=None, n_out=None, final=False):
        if layer is None:
            p = self.dropout_rate
            if n_out is None:
                n_out = self.layer_size
            if final:
                n_out = self.data['y'].shape[1]     
                act = None
                p = 0
            else:
                act = self.default_act
            if len(self.layers) == 0:
                input_ = self.tv['X']
                input_sh = (self.batch_size, self.data['X'].shape[1])
                p = self.dropout_rate_1
            else:
                input_ = self.layers[-1].output
                input_sh = self.layers[-1].output_sh
            layer = FullyConnectedLayer(self.rng, input_, input_sh=input_sh, 
                                        n_out=n_out, activation=act, 
                                        name='layer%d' % len(self.layers), p=p)
        super().add_layer(layer)