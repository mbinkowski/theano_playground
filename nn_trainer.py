# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:36:59 2016

@author: mbinkowski
"""
from nn_class import *
from IPython import display

class Trainer(object):
    def __init__(self, nn, dump_file='nn_results.pickle'):
        self.nn = nn
        self.nn.trainer = self
        self.dump_file = dump_file
        self.iters = [0]
        self.improvement = [1] * 5
        self.secondary = 'R2' if (nn.type == 'regressor') else 'accuracy'
        self.monitor_activations = False
        self.loss_history = {
            'train': [],
            'train ' + self.secondary: [],
            'valid': [],
            'valid ' + self.secondary: [],
            'test' : [],
            'test ' + self.secondary: [],
            'learning_rate': []
        }
        self.last_valid_loss = np.inf
        self.best_losses = {
            'valid': np.inf, 
            'test': np.inf, 
            'test R2': -np.inf, 
            'iteration': 0,
            'epoch': 0,
            'params': []
        }
        self.time0 = time.time()
        
        
    def train_function(self, param_values):
        param_values, _ = self.nn._set_params(param_values)
        cost = self.nn.tf['train']['model'](self.iters[0] % self.nn.tf['train']['batches'])
        return np.mean(cost)
    
    def grad_function(self, param_values):
        grad_val = self.nn.tf['grad']['model'](self.iters[0] % self.nn.tf['grad']['batches'])
        if any([np.isnan(g).any() for g in grad_val]):
            print('nans at grad_val stage')#, returning 0 gradient.')
        flat_grad_val = np.concatenate([g.flatten() for g in grad_val]).clip(-1e4, 1e4)
        norm = np.sqrt((flat_grad_val**2).sum())
        div = norm.clip(1000.0, np.inf)/1000.0
        flat_grad_val /= div
        if ((self.iters[0] % self.nn.tf['train']['batches']) == 0) and self.print_grad:
            print('|grad|^2 = %.5f, max |grad| = %.5f' % (norm, flat_grad_val.max()))
        return flat_grad_val        

    def callback(self, param_values, message=''):
        self.iters[0] += 1
        param_values, grad_values = self.nn._set_params(param_values)
    #     print('check_grad: ' + repr(check_grad(train_function, grad_function, param_values)))
        if (self.iters[0] % self.nn.tf['train']['batches'] != 0) and (self.iters[0] != int(.2 * self.nn.batches['train'])):
            self.dropout('on')
            return param_values, grad_values, 1#(epoch - self.improvement[0] > 100)
        
        self.dropout('off')
        if self.interactive_display:
            plt.gca().cla()
            display.clear_output(wait=True)
            
        print(message)
        print('|params|^2 = %.2f, max |params| = %.2f' % 
            (sum([(p.get_value()**2).sum() for p in self.nn.params]), 
             np.max([np.max(np.abs(p.eval())) for p in self.nn.params])))
    #     print('params[0] ' + repr(params[1].eval()))
        loss, _, __= self._loss_stats('train_loss')
        loss, _, __ = self._loss_stats('valid')
        
        self.improvement.append(loss < self.best_losses['valid'])#self.last_valid_loss)
        self.last_valid_loss = loss
        
        if loss < self.best_losses['valid']:
            self.best_losses['valid'] = loss
            self.best_losses['test'], self.best_losses['mean test R2'], self.best_losses['test R2'] = self._loss_stats('test')
            self.best_losses['params'] = self.nn.get_param_values()
            self.best_losses['iter'] = self.iters[0]
            self.best_losses['epoch'] = self.iters[0] // self.nn.tf['train']['batches'] + 1
            with open(self.dump_file, 'wb') as f:
                pickle.dump([self.best_losses, self.iters[0]], f)

        if self.interactive_display:
            self.plot()
        
        self.callback2(self.nn)
        
    #         self.improvement[0] = epoch
        ru = min(1.005, (np.mean(self.improvement[-3:]) + 6.4)/7) * (1 - (sum(self.improvement[-self.early_stop:]) == 0))
        self.loss_history['learning_rate'].append(self.loss_history['learning_rate'][-1] * ru)
        self.dropout('on')
        return param_values, grad_values, ru#False

    def _loss_stats(self, mod):
        epoch = self.iters[0] // self.nn.tf['train']['batches']
        elapsed_time = time.time() - self.time0
#        print(epoch)
#        print(elapsed_time)
#        print(self.nn.tf[mod]['name'])
        mess = self.nn.tf[mod]['name'] + (': epoch %d, iter %d, time %.1f' % (\
                                           epoch, self.iters[0], elapsed_time))
        loss, loss2, losses = self.nn.type_loss_stats(self.nn.tf[mod], mess)
        name = self.nn.tf[mod]['name'].split('_')[0]
        self.loss_history[name].append(loss)
        self.loss_history[name + ' ' + self.secondary].append(loss2)
        return loss, loss2, losses
        
    def plot(self):
        plt.gca().cla()
        plt.close()
        fig = plt.figure(1, (13, 9))
        axes_no = 2 + len(self.nn.layers) * self.monitor_activations
        axes = [fig.add_subplot(int(np.ceil(axes_no/3)), 
                                2 + self.monitor_activations, 
                                i + 1) for i in np.arange(axes_no)]
        for ax, lossn in zip(axes[:2], ['', ' ' + self.secondary]):
            xx = np.arange(len(self.loss_history['train'])) 
            ax.scatter(xx, self.loss_history['train' + lossn], label='train', marker='o')
            ax.scatter(xx, self.loss_history['valid' + lossn], label='valid', marker='x')
            ax.set_xlabel('epoch')
            ax.set_xticks(xx[::int(max(1, len(xx)/10, np.sqrt(len(xx)/2.5)))])
            ax.set_ylabel(('loss' + lossn).split(' ')[-1])
            ax.set_xlim(-.5, len(xx) - .5)
            ax.set_title(('loss' + lossn).split(' ')[-1])

        if self.monitor_activations:
            sample = 20
            for ax, f, l in zip(axes[2:], self._activation_tf, self.nn.layers):
                act = np.concatenate([f(i) for i in np.arange(sample)])
                ax.imshow(act.transpose(), vmin=0, vmax=max(1, act.max()), 
                          aspect=l.output_sh[0]*sample/(l.output_sh[1] * 2),
                          interpolation='nearest')     
                ax.set_title(' '.join(repr(l).split(' ')[:2]))
                ax.grid('off')
        
#        fig.tight_layout()    
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc = (0.40, 0.02), ncol=2, fontsize=12)
        display.display(plt.gcf())
    
    def NoneCallback(self, _):
        pass
    
    def dropout(self, key='on'):
        if key == 'on':
            for il, l in enumerate(self.nn.layers):
                if hasattr(l, 'dropout_on'):
                    l.dropout_on()
        else:
            for l in self.nn.layers:
                if hasattr(l, 'dropout_off'):
                    l.dropout_off()
    
    def train(self, start_rate=.005, disp=.005, gtol=1e-5, maxiter=3e10, 
              beta=.4, tau=.7, ftol=.8, early_stop=10, print_grad=True,
              interactive_display=True, callback2=None, monitor_activations=True):
        print('Training network. Start time %.2f' % (time.time() - self.time0))
        if callback2 is None:
            self.callback2 = self.NoneCallback
        else:
            self.callback2 = callback2
            
        if monitor_activations:
            self._define_activation_tf()
            self.monitor_activations = True
        self.print_grad = print_grad
        self.loss_history['learning_rate'].append(start_rate)
        self.early_stop = early_stop
        self.interactive_display = interactive_display
        if interactive_display:
            plt.style.use('ggplot')
            plt.ion()
  
        from my_cg import my_fmin_cg
        my_fmin_cg(
            f=self.train_function,
            x0=np.concatenate([p.get_value().flatten() for p in self.nn.params]),
            fprime=self.grad_function,
            callback=self.callback,
            disp=disp,
            gtol=gtol,
            maxiter=int(maxiter),
            beta=beta,
            tau=tau,
            ftol=ftol,
            start_rate=start_rate
        )
        
        self.best_losses.update({
            'layers': len(self.nn.layers),
            'X.shape': self.nn.data['X'].shape,
            'total_iters': self.iters[0],
            'total_epochs': int(self.iters[0] / self.nn.batches['train'])
        })
        print('Training network finished. Time %.2f' % (time.time() - self.time0))
        return self.best_losses['params'], self.loss_history
        
    def _define_activation_tf(self):
        self._activation_tf = []
        i, bs = self.nn.tv['idx'], self.nn.batch_size
        for l in self.nn.layers:
            self._activation_tf.append(theano.function(
                [i], 
                l.output, 
                givens={self.nn.tv['X']: self.nn.shared['X'][bs * i : bs * (1 + i)]}
                ))