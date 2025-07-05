from copy import deepcopy

import numpy as np
import torch
from fontTools.varLib.avarPlanner import WEIGHTS
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt




def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, noise_multiplier_list,minibatch_size, microbatch_size,*args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier

            self.noise_multiplier_list = noise_multiplier_list

            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.grad_batch_histo = []

            #self.WEIGHT =[1.0 , 1.0 , 1.0 ,2.0 , 0.86 , 2.0 , 1.0]
            self.WEIGHT = [2.0 , 1.0 , 1.0 ,2.0 , 1.0 , 1.0 , 2.0]
            self.count = [0,0,0,0,0,0,0]
            self.rho = [0.1016, 0.1562, 0.1562, 0.0781, 0.2344, 0.1953, 0.0781]


            """for ADADP"""
            self.p0 = None
            self.p1 = None
            self.accepted = 0
            self.failed = 0
            self.lrs_history = []
            self.lrs = 1e-3


            for id, group in enumerate(self.param_groups):
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in
                                        group['params']]


            temp = copy.deepcopy(self.param_groups)
            self.myparam_groups = {i: temp for i in range(7)}
            #print(type(self.myparam_groups))

        """----------estimate clip by histogram--------------------------- """

        def clipping_estimate(self,cat):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            #if(int(cat)==3 or int(cat)==6 or int(cat)==0 or int(cat)==1):
            self.grad_batch_histo.append(total_norm)


        def compute_clip(self,p):
            """plt.hist(self.grad_batch_histo, bins=50, color='blue', alpha=0.7)
            plt.title('Histogram of Gradient Batch')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()"""


            clip_value = np.percentile(self.grad_batch_histo, p)
            self.grad_batch_histo = []
            return clip_value

        """ ---------------------zero gradient-------------------------"""

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def zero_accum_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def zero_accum_grad_Hn(self):

            self.count =[0,0,0,0,0,0,0]
            for i in range(7):
                for group in self.myparam_groups[i]:
                    for accum_grad in group['accum_grads']:
                        if accum_grad is not None:
                            accum_grad.zero_()

            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()
        """-----------------------micro step of all-----------------------------"""

        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5
            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm

        def microbatch_step_AutoDP(self,w):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.

            total_norm = total_norm ** .5

            self.l2_norm_clip = total_norm*w if total_norm*w>20.0 and total_norm*w<40  else 20

            clip_coef = min(self.l2_norm_clip / (total_norm+ 1e-6), 1.)

            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))

            return total_norm



        """---------------step-------------------------------------------"""
        def step_dp(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)

        def step_dp_Hn(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        clip_coef = max(self.WEIGHT)
                        #param.grad.data.add_(clip_coef * self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        for i in range(7):
                            if (self.count[i] > 0):
                                param.grad.data.add_(self.WEIGHT[
                                    i] * self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        #(self.count[i] / self.minibatch_size)
                        #param.grad.data.add_( clip_coef* self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size/self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)

        def step_DDP(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()

                        for i in range(7):
                            param.grad.data.add_((1/(self.rho[i]*self.minibatch_size))* self.l2_norm_clip * self.noise_multiplier_list[i] * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(1/7)



        def step_dp_agd(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'],
                                             group['accum_grads']):
                    if param.requires_grad:

                        param.grad.data = accum_grad.clone()

                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))

                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)


        def step_adadp_step1(self,*args, **kwargs):
            del self.p0
            self.p0 = []

            del self.p1
            self.p1 = []

            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is None:
                        continue

                    dd = p.data.clone()
                    self.p0.append(dd)

                    self.p1.append(p.data - self.lrs * p.grad.data)
                    p.data.add_(-0.5 * self.lrs, p.grad.data)

        def step_adadp_step2(self, tol=1.0 ,*args, **kwargs):
            for group in self.param_groups:

                err_e = 0.0

                for ijk, p in enumerate(group['params']):
                    p.data.add_(-0.5 * self.lrs, p.grad.data)
                    err_e += (((self.p1[ijk] - p.data) ** 2 / (
                        torch.max(torch.ones(self.p1[ijk].size()).cuda(), self.p1[ijk] ** 2))).norm(1))

                err_e = np.sqrt(float(err_e))

                self.lrs = float(self.lrs * min(max(np.sqrt(tol / err_e), 0.9), 1.1))

                ## Accept the step only if err < tol.
                ## Can be sometimes neglected (more accepted steps)
                if err_e > 1.0 * tol:
                    for ijk, p in enumerate(group['params']):
                        p.data = self.p0[ijk]
                if err_e < tol:
                    self.accepted += 1
                else:
                    self.failed += 1

                self.lrs_history.append(self.lrs)



    return DPOptimizerClass

DPAdam_Optimizer = make_optimizer_class(Adam)
DPAdagrad_Optimizer = make_optimizer_class(Adagrad)
DPSGD_Optimizer = make_optimizer_class(SGD)
DPRMSprop_Optimizer = make_optimizer_class(RMSprop)

def get_dp_optimizer(name,lr,C_t,sigma,sigma_list = None,batch_size=128 ,microbatch_size=1,model=None):

    if name =='adam':
        optimizer = DPAdam_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,

            noise_multiplier_list = sigma_list,

            minibatch_size=batch_size,
            microbatch_size=microbatch_size,
            params=model.parameters(),
            lr=lr,
            betas=(0.9, 0.99),
            weight_decay=0.0005
        )
    else:
        optimizer = DPSGD_Optimizer(
            l2_norm_clip=C_t,
            noise_multiplier=sigma,

            noise_multiplier_list = sigma_list,

            minibatch_size=batch_size,
            microbatch_size=1,
            params=model.parameters(),
            lr=lr,
            momentum=0
        )
    return optimizer
