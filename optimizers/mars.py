# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/AGI-Arena/MARS
import math
import torch
from torch.optim.optimizer import Optimizer

def exists(val):
    return val is not None

def update_fn(p, grad, exp_avg, exp_avg_sq, lr, wd, beta1, beta2, last_grad, eps, amsgrad, max_exp_avg_sq, step, gamma,
              mars_type, is_grad_2d, optimize_1d, lr_1d_factor, betas_1d, weight_decay_1d):
    """
    Performs a single optimization step for MARS optimizer.
    """
    # Compute corrected gradient using MARS variance reduction
    mars_factor = gamma * beta1 / (1 - beta1)
    c_t = (grad - last_grad).mul(mars_factor).add(grad)
    
    # Apply optimizer based on mars_type
    if mars_type == "mars-adamw":
        # AdamW update
        if not is_grad_2d or optimize_1d:
            # Use 1D hyperparameters for 1D parameters
            if not is_grad_2d and not optimize_1d:
                current_beta1, current_beta2 = betas_1d
                current_lr = lr * lr_1d_factor
                current_wd = weight_decay_1d
            else:
                current_beta1 = beta1
                current_beta2 = beta2
                current_lr = lr
                current_wd = wd
        else:
            current_beta1 = beta1
            current_beta2 = beta2
            current_lr = lr
            current_wd = wd
            
        exp_avg.mul_(current_beta1).add_(c_t, alpha=1 - current_beta1)
        exp_avg_sq.mul_(current_beta2).add_(c_t.square(), alpha=1 - current_beta2)
        
        if amsgrad:
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(eps)
        else:
            denom = exp_avg_sq.sqrt().add_(eps)
            
        bias_correction1 = 1 - current_beta1 ** step
        bias_correction2 = 1 - current_beta2 ** step
        step_size = current_lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        
        p.mul_(1 - current_lr * current_wd)
        p.addcdiv_(exp_avg, denom, value=-step_size / bias_correction2_sqrt)
        
    elif mars_type == "mars-lion":
        # Lion update
        if not is_grad_2d or optimize_1d:
            if not is_grad_2d and not optimize_1d:
                current_beta1, current_beta2 = betas_1d
                current_lr = lr * lr_1d_factor
                current_wd = weight_decay_1d
            else:
                current_beta1 = beta1
                current_beta2 = beta2
                current_lr = lr
                current_wd = wd
        else:
            current_beta1 = beta1
            current_beta2 = beta2
            current_lr = lr
            current_wd = wd
            
        update = exp_avg.mul(current_beta1).add_(c_t, alpha=1 - current_beta1).sign_()
        p.mul_(1 - current_lr * current_wd)
        p.add_(update, alpha=-current_lr)
        exp_avg.mul_(current_beta2).add_(c_t, alpha=1 - current_beta2)
        
    return exp_avg, exp_avg_sq

class MARS(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=0., amsgrad=False, gamma=0.025, 
                 is_approx=True, mars_type="mars-adamw", optimize_1d=False, lr_1d=3e-3, betas_1d=(0.9, 0.95), weight_decay_1d=0.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        assert mars_type in ["mars-adamw", "mars-lion", "mars-shampoo"], "MARS type not supported"
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, 
                        mars_type=mars_type, gamma=gamma, 
                        optimize_1d=optimize_1d, weight_decay_1d=weight_decay_1d)
        super(MARS, self).__init__(params, defaults)
        self.eps = eps
        self.update_fn = update_fn
        self.lr = lr
        self.weight_decay=weight_decay
        self.amsgrad = amsgrad
        self.step_num = 0
        self.is_approx = is_approx
        self.gamma = gamma
        self.mars_type = mars_type
        self.optimize_1d = optimize_1d
        self.lr_1d_factor = lr_1d / lr
        self.weight_decay_1d = weight_decay_1d
        self.betas_1d = betas_1d

    @torch.no_grad()
    def update_last_grad(self):
        """
        Update last gradient for MARS variance reduction.
        Should be called after optimizer.step() in training loop.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'last_grad' in state:
                    state['last_grad'].copy_(p.grad)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            mars_type = group['mars_type']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                
                state = self.state[p]
                # State initialization
                if len(state) <= 1:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Last Gradient
                    state['last_grad'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                else:
                    max_exp_avg_sqs.append(0)
                    
                state['step'] += 1
                state_steps.append(state['step'])
            
            # Perform optimization step
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                max_exp_avg_sq = max_exp_avg_sqs[i]
                step = state_steps[i]
                state = self.state[param]
                last_grad = state['last_grad']
                lr, wd = group['lr'], group['weight_decay']
                
                is_grad_2d = (len(grad.shape) == 2)
                exp_avg, exp_avg_sq = self.update_fn(
                    param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    last_grad,
                    self.eps,
                    amsgrad,
                    max_exp_avg_sq,
                    step,
                    gamma,
                    mars_type=self.mars_type,
                    is_grad_2d=is_grad_2d,
                    optimize_1d=self.optimize_1d,
                    lr_1d_factor=self.lr_1d_factor,
                    betas_1d=self.betas_1d,
                    weight_decay_1d=self.weight_decay if self.optimize_1d else self.weight_decay_1d
                )
                if self.is_approx:
                    state['last_grad'] = grad.clone()
        
        self.step_num = max(state_steps) if state_steps else 0
        
        return loss
