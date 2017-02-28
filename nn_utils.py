import theano
import numpy as np
import time
from theano.compile.nanguardmode import NanGuardMode
import theano.tensor as T

def ev_model(tv, shared, data, batch_size, Loss, start, batches, name):
#    left = tv['idx'] * batch_size + start
    model = theano.function(
        [tv['idx']],
        Loss,
        givens=dict([(tv[k], shared[k][tv['idx'] * batch_size + start: 
            (tv['idx'] + 1) * batch_size + start]) for k in shared]),
        name=name,
        on_unused_input='ignore'
#        ,mode=NanGuardMode(True,True,True)
    )
    _data = data['y'][start: start + batches * batch_size, :]
    _mask = data['mask'][start: start + batches * batch_size, :]
    _var = (((_data - _data.mean(axis=0))*_mask)**2).sum(axis=0)#/valid_mask.sum(axis=0).clip(1, np.inf)
#    print(name + '_var: ' + repr(_var))
    return {'model': model, 'var': _var, 'mask': _mask, 'batches': batches, 
            'name': name}
    

def reg_cost(params, L1_reg, L2_reg):
    L1 = T.sum([abs(p).mean() for p in params])
    L2 = T.sum([(p ** 2).mean() for p in params])
    return (L1_reg * L1 + L2_reg * L2)/len(params)

   
def l2_loss(nn, l1=0.0, l2=0.0):
    Losses = ((nn.layers[-1].output - nn.tv['y']) * nn.tv['mask'])**2
    Loss = (Losses.sum(axis=0)/nn.tv['mask'].sum(axis=0).clip(1, np.inf)).mean()
#    tvmean = (nn.tv['mask'] * nn.tv['y']).sum(axis=0)/nn.tv['mask'].sum(axis=0).clip(1, np.inf)
#    LossR2 = (Loss_per_dim/(((nn.tv['y'] - tvmean) * nn.tv['mask'])**2).sum(axis=0).clip(1, np.inf)).mean()
    Cost = Loss + reg_cost(nn.params, l1, l2)
    return (Losses, Cost)


def cross_entropy_loss(nn, l1=0.0, l2=0.0, th=.5):
    out = nn.layers[-1].output.clip(1e-10, 1 - 1e-10)
    groundtruth = T.cast(nn.tv['y'] > 0, dtype=theano.config.floatX)#5 * (T.sgn(nn.tv['y']-1e-9) + 1)
    Losses = - (groundtruth * T.log(out) + (1 - groundtruth) * T.log(1 - out)) * nn.tv['mask']
    Loss = (Losses.sum(axis=0)/nn.tv['mask'].sum(axis=0).clip(1, np.inf)).mean()
    Cost = Loss + reg_cost(nn.params, l1, l2)
    pred = .5 * (T.sgn(out - th) + 1)
    tpfn = ((1 - groundtruth) * (1 - pred) + groundtruth * pred ) * nn.tv['mask']
    return ([Losses, tpfn], Cost)


def cross_entropy_lossN(nn, l1=0.0, l2=0.0):
    out = nn.layers[-1].output.clip(1e-10, 1 - 1e-10)
    Losses = - (nn.tv['y'] * T.log(out)).sum(axis=2) * nn.tv['mask']
    Loss = (Losses.sum(axis=0)/nn.tv['mask'].sum(axis=0).clip(1, np.inf)).mean()
    Cost = Loss + reg_cost(nn.params, l1, l2)
    pred = T.cast(out == T.argmax(out, keepdims=True), dtype=theano.config.floatX)
    tpfn = (nn.tv['y'] * pred).sum(axis=2) * nn.tv['mask']
    return ([Losses, tpfn], Cost)

    

def regressor_loss_stats(model, message=''):
    losses = np.array([model['model'](idx) for idx in np.arange(model['batches'])])
    if len(losses.shape) == 1:
        losses = [losses.sum(axis=0)]
    while len(losses.shape) >= 2:
#        print(losses.shape)        
        losses = losses.sum(axis=0)
#    print(losses.shape)        
#     print(np.array(valid_losses).sum(axis=0)/data['mask'][n_train: n_train + n_valid].sum(axis=0))
    if any(np.isnan(losses)):
         print('NAN!!!!!!!!!!')
    loss = (losses/model['mask'].sum(axis=0)).mean()
    R2 = 1 - (losses/model['var']).mean()
    print('Regressor ' + message + (', loss = %f, R2 = %.4f' % (loss, R2)))
    R2s = 1 - losses/model['var']
    print(R2s)
    return loss, R2, R2s
    
def classifier_loss_stats(model, message=''):
    l0 = [model['model'](idx) for idx in np.arange(model['batches'])]
#    print(l0)
    losses = np.array([l[0] for l in l0])
    accuracies = np.array([l[1] for l in l0])
    print(losses.shape)
    if len(losses.shape) == 1:
        losses = np.array([losses.sum(axis=0)])
        accuracies = np.array([accuracies.sum(axis=0)])
    while len(losses.shape) >= 2:
#        print(losses.shape)        
        losses = losses.sum(axis=0)
        accuracies = accuracies.sum(axis=0)
#    print(losses.shape)        
#     print(np.array(valid_losses).sum(axis=0)/data['mask'][n_train: n_train + n_valid].sum(axis=0))
    if any(np.isnan(losses)):
         print('NAN!!!!!!!!!!')
    loss = (losses/model['mask'].sum(axis=0)).mean()
    accuracies = accuracies/model['mask'].sum(axis=0)
    accuracy = accuracies.mean()
    print('Classifier ' + message + (', accurracy = %f, loss = %f' % (accuracy, loss)))
    print(accuracies)
    return loss, accuracy, accuracies
    
    
def callback2(nn):
    for i, l in enumerate(nn.layers):
        mess = 'layer (%d) params range: ' % i
        for p in l.params:
            pars = p.eval()
            mess += '[%.2f, %.2f] ' % (pars.min(), pars.max())
        if len(l.params) > 0:
            print(mess)
#     layer2act = np.concatenate([f(i) for i in range(50)])
#     plt.imshow(layer2act.transpose(), vmin=0, vmax=1, aspect=10)
        
        