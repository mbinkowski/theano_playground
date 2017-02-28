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
rng = np.random.RandomState(21312)

def nn_def_train(params):
    layer_no, layer_size, dropout_rate, dropout_first, start_learning_rate, max_norm = params
    max_iterations=200000
    last_improvement = 0
    
    with open('preprocessed_nn3_freq10_ahead60.pickle', 'rb') as f:
        XX, XXmean, XXstd, yy, yymean, yystd, sources = pickle.load(f)
    # iii = np.arange(0,5500)[343::90]
    # XX = XX[iii]
    # yy = yy[iii]
    batch_size = 32
    train_share, valid_share = .8, .1
    batches_train, batches_valid = int((XX.shape[0] * train_share) // batch_size), int((XX.shape[0] * valid_share) // batch_size)
    batches_test = int(XX.shape[0] // batch_size) - batches_train - batches_valid
    n_train, n_valid, n_test = batches_train * batch_size, batches_valid * batch_size, batches_test * batch_size
    order = np.r_[np.random.permutation(n_train), np.random.permutation(np.arange(n_train, XX.shape[0]))]
    dir = 0
    
    data = {
        'X': np.transpose(XX[order, :, :, :, :], (0, 1, 4, 2, 3)),
        'y': yy[order, :1, dir, 0],
        'mask': np.ones(yy[order, :1, dir, 1].shape)
    }
    
    var = ((data['y'][n_train + n_valid:] - data['y'][n_train + n_valid:].mean())**2).mean()
    print('Zero prediction R2 = %.2f' % (1 - (data['y'][n_train + n_valid:]**2).mean()/var))
    print('Mean prediction R2 = %.2f' % (1 - ((data['y'][n_train +  n_valid:] - data['y'][:n_train].mean())**2).mean()/var))
    print('Last return prediction R2 = %.2f' % 
          (1 - ((data['y'][n_train + n_valid:] - data['X'][n_train + n_valid:, :, 0, dir, 0])**2).mean()/var))

    data['X'] = data['X'][:, :, :1, :, :] # leave just 1st lag
#    data['X'] = np.transpose(data['X'].reshape((XX.shape[0], XX.shape[1], XX.shape[4], np.prod(XX.shape[2:4]))),
#                             (0, 2, 3, 1))
    ### [obs] x [price/time] x [lags x dir] x [mms]
    data['X'] = data['X'][:, 0, :, :] # leave just price
    ### [obs] x [price] x [lags x dir] x [mms]
    
    
    #####TEST
    # import copy
    # data['y'] = copy.deepcopy(data['X'][:, 0, 0, :2])
    # data['mask'] = data['mask'][:, :2]
    # data['X'] = copy.deepcopy(data['X'][:, 0, 0, :4])#
    
    data['X'] = data['X'].reshape((data['X'].shape[0], np.prod(data['X'].shape[1:])))
    ### [obs] x [{price} x lags x dir x mms]
    data['y'] = data['X'][:, :1]
    tv = {
        'X': T.matrix(name='X'),#T.tensor4(name='X'),
        'y': T.matrix(name='y'),
        'mask': T.matrix(name='mask'),
        'idx': T.lscalar(name='idx')
        }
    
    shared = dict([(k, theano.shared(np.asarray(v, dtype=theano.config.floatX), borrow=True)) for k, v in data.items()])


    act = T.nnet.relu
    layers = [FullyConnectedLayer(
        rng, tv['X'], 
        input_sh=(batch_size, data['X'].shape[1]), 
        n_out=layer_size, activation=act, name='regression', 
        p=dropout_first
        )]
    for i in np.arange(layer_no - 2):
        i = int(i)
        layers.append(
            FullyConnectedLayer(rng, layers[i].output, 
                                input_sh=layers[i].output_sh, 
                                n_out=layer_size, 
                                activation=act, 
                                name='regression', 
                                p=dropout_rate)
            )
    layers.append(
        FullyConnectedLayer(rng, layers[-1].output, 
                            input_sh=layers[-1].output_sh, 
                            n_out=data['y'].shape[1], 
                            activation=None, 
                            name='regression', 
                            p=0)
        )
        
    layers[0].pretrain(shared['X'], batch_size, batches_train, rate=.05, rate_min=.03, steps=max_iterations, use_b=True)
    for layerB in layers[1:]:
        pretrained_B = layerB.get_pretrain_sample(tv['X'], tv['idx'], shared['X'], batch_size, batches_train)
        layerB.pretrain(pretrained_B, batch_size, batches_train, rate=.05, rate_min=.01, steps=max_iterations, use_b=False)
        
    params = []
    for l in layers:
        params += l.params
        
    Loss_per_dim = (((layers[-1].output - tv['y']) * tv['mask'])**2).sum(axis=0)
    Loss = (Loss_per_dim/tv['mask'].sum(axis=0).clip(1, np.inf)).mean()
    tvmean = (tv['mask'] * tv['y']).sum(axis=0)/tv['mask'].sum(axis=0).clip(1, np.inf)
    LossR2 = (Loss_per_dim/(((tv['y'] - tvmean) * tv['mask'])**2).sum(axis=0).clip(1, np.inf)).mean()
    Cost = Loss + reg_cost(params, 0.0, 0.0)
    
    # ff = theano.function([tv['X']], layer.output, on_unused_input='warn')
    # print(shared['X'][:25,:,:,:].eval().shape)
    # print(ff(data['X'][:25,:,:,:]).shape)
    
    # grads = [g.clip(-1e6, 1e6) for g in T.grad(Cost, params)]
    grads = [g for g in T.grad(Cost, params)]
    
    ### Defining theano models
    valid = ev_model(tv, shared, data, batch_size, 
                     Loss_per_dim, n_train, batches_valid, 'valid')
    
    test = ev_model(tv, shared, data, batch_size, 
                    Loss_per_dim, n_train + n_valid, batches_test, 'test')
    
    train = ev_model(tv, shared, data, batch_size,
                     Cost, 0, batches_train, 'train')
    
    train_loss = ev_model(tv, shared, data, batch_size,
                     Loss_per_dim, 0, batches_train, 'train_loss')
    
    grad = ev_model(tv, shared, data, batch_size,
                   grads, 0, batches_train, 'grad')
    print('Network ready')
#    
#    for p, v in zip(params, pretrained_params):#best_losses['params']):#
#        p.set_value(v)
    
    # Training variables
    best_losses = {'valid': np.inf, 'test': np.inf, 'test R2': -np.inf, 'params': []}
    cum_sizes = np.cumsum([0] + [np.prod(p.get_value().shape) for p in params])
    
    
    def set_params(flat_param_values):
        if any(np.isnan(flat_param_values).ravel()):
            print('nans passed to set_params')
        scaled = False
        new_params, scaled_params = [], []
        scaled_flat_param_values = copy.deepcopy(flat_param_values)
        for i, p in enumerate(params):
            sh = p.get_value().shape
            l, r = int(cum_sizes[i]), int(cum_sizes[i+1])
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
            div2 = (np.sqrt(div/max_norm)).clip(1, np.inf)
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
            flat_grad_values = -grad_function(flat_param_values)
            if not any(np.isnan(flat_grad_values)):
                for p, v in zip(params, scaled_params):
                    p.set_value(v)
    #             print('rescaling')
                return scaled_flat_param_values, flat_grad_values 
        for p, v in zip(params, new_params):
            p.set_value(v)
        return flat_param_values, None
    
    def train_function(param_values):
        for l in layers:
            l.dropout_on()
    #     layer2P.dropout_on()
    #     layer.dropout_on()
        param_values, _ = set_params(param_values)
        cost = train['model'](iters[0] % train['batches'])
        return np.mean(cost)
    
    print_grad = True
    def grad_function(param_values):
    #     param_values = set_params(param_values)
        grad_val = grad['model'](iters[0] % grad['batches'])
        if any([np.isnan(g).any() for g in grad_val]):
            print('nans at grad_val stage')#, returning 0 gradient.')
    #         print('grad_val = ' + repr(grad_val))    
    #         return np.concatenate([np.zeros(g.flatten().shape) for g in grad_val])
    #         raise ValueError('nans at grad_val stage')
    
    #     if iters[0] % valid_freq != 0:
    #         for i, g in enumerate(grad_val):
    #             print((i, g))
        flat_grad_val = np.concatenate([g.flatten() for g in grad_val]).clip(-1e4, 1e4)
        norm = np.sqrt((flat_grad_val**2).sum())
        div = norm.clip(1000.0, np.inf)/1000.0
        flat_grad_val /= div
        if ((iters[0] % valid_freq) == 0) and print_grad:
            print('|grad|^2 = %.5f, max |grad| = %.5f' % (norm, flat_grad_val.max()))
    #     if any(np.isnan(flat_grad_val)):
    #         print('nans at flat_grad_val stage')
    #         print('grad_val = ' + repr(grad_val))
    #     print(flat_grad_val.shape)
        return flat_grad_val
    
    from scipy.optimize import check_grad
    
    last_improvement = [0]
    def callback(param_values, message=''):
        iters[0] += 1
        epoch = iters[0] // batches_train
        param_values, grad_values = set_params(param_values)
    #     print('check_grad: ' + repr(check_grad(train_function, grad_function, param_values)))
        if iters[0] % valid_freq != 0:
            return param_values, grad_values, (epoch - last_improvement[0] > 20)
        for l in layers:
            l.dropout_off()
        print(message)
        print('|params|^2 = %.2f, max |params| = %.2f' % (sum([(p.get_value()**2).sum() for p in params]), 
                                                          np.max([np.max(np.abs(p.eval())) for p in params])))
    #     print('params[0] ' + repr(params[1].eval()))
        loss, _ = loss_stats(train_loss, time0, iters[0], epoch)
        loss, _ = loss_stats(valid, time0, iters[0], epoch)
        if loss < best_losses['valid']:
            best_losses['valid'] = loss
            best_losses['test'], best_losses['test R2'] = loss_stats(test, time0, iters[0], epoch)
            best_losses['params'] = [p.eval() for p in params]
            with open('results_nn_3_1.pickle', 'wb') as f:
                pickle.dump([best_losses, iters[0]], f)
            last_improvement[0] = epoch
        return param_values, grad_values, False
    
    # Training parameters
    epoch = 0
    iters = [0]
    max_epochs = 100
    time0 = time.time()
    valid_freq = 200
    
    max_iters = max_iterations
    beta = .6
    tau = .7
    ftol = .8
            
    import scipy.optimize
    import my_scipy_optimize
    imp.reload(my_scipy_optimize)
    from my_scipy_optimize import fmin_cg
    import my_cg
    imp.reload(my_cg)
    from my_cg import my_fmin_cg
    print ("Optimizing using fmin_cg...")
    best_params = my_fmin_cg(
        f=train_function,
        x0=np.concatenate([p.get_value().flatten() for p in params]),
        fprime=grad_function,
        callback=callback,
        disp=0.005,
        gtol=10**(-5),
        maxiter=max_iters,
        beta=beta,
        tau=tau,
        ftol=ftol,
        start_rate=start_learning_rate
    )
    print('Finished, time = ' + repr(time.time() - time0))
    best_losses.update({'layers' : layers})
    return best_losses