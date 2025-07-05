import argparse
import torch
import torch.nn as nn

import torchvision.models as models

from rdp_accountant import get_privacy_spent,compute_rdp,get_sigma,get_sigma_list,get_sigma_Auto

from HAM10000_7class import getSkinDatasetBalanced

from dp_optimizer import get_dp_optimizer


from algorithm.DP import DP

from algorithm.SAD_DPSGD import SAD_DPSGD


from algorithm.AutoDP_L import AutoDP
def group_norm(num_channels):
    return nn.GroupNorm(num_groups=32, num_channels=num_channels)

def DPHAM10000(args):
    best_acc=0
    print('===================================================================================================')
    print('===================================================================================================')
    print('===================================================================================================')
    print('==> Preparing data..')
    ## preparing data for training && testing

    batch_size = args.batch_size

    train_loader,val_loader,training_set,test_set,_=getSkinDatasetBalanced(batch_size)


    n_training = len(training_set)
    n_test = len(test_set)
    print('# of training examples: ', n_training, '# of testing examples: ', n_test)




    noise_multiplier_list = []




    sampling_prob = (args.batch_size / n_training)
    steps = int(args.n_epoch / sampling_prob)
    print(steps)
    print(sampling_prob)



    if(args.sigma==None and args.eps!=None):


        if(args.Method =='SAD-DPSGD' or args.Method =='Auto-DP-SGD-S'):
            sigma,eps = get_sigma_list(float(sampling_prob), steps, args.eps,  args.delta,args.R ,args.qR, args.numOfstep)
        elif(args.Method == 'Auto-DP-SGD-L'):
            sigma, eps = get_sigma_Auto(float(sampling_prob), steps, args.eps, args.delta)
        else:
            sigma, eps = get_sigma(float(sampling_prob), steps, args.eps, args.delta)

        noise_multiplier = sigma

    elif(args.sigma!=None and args.eps==None):
        max_order = 64
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, max_order + 1))
        rdp = steps * compute_rdp(sampling_prob, args.sigma, 1, orders)
        eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=args.delta)
        noise_multiplier = args.sigma
    else:
        print("give eps, or sigma only")

    if(args.private):
        print('noise scale: ', noise_multiplier, 'privacy guarantee: ', eps)

    print('\n==> Strat training')


    net = models.resnet50(pretrained=True) #models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    net.fc = nn.Linear(in_features=2048, out_features=7)


    net.cuda()


    num_params = 0
    for p in net.parameters():
        num_params += p.numel()

    weights = [0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2]


    class_weights = torch.FloatTensor(weights).cuda()


    if(args.private):
        criterion = nn.CrossEntropyLoss(weight=class_weights,reduction='sum')
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)


    if(args.Method == 'DP-SGD'):
        trainer = DP(args)
        optimizer = get_dp_optimizer('SGD', args.lr, args.clip, noise_multiplier, noise_multiplier_list,args.batch_size,
                                     1,net)
        trainer.train(net, train_loader, val_loader, optimizer, criterion, args)
    elif(args.Method == 'SAD-DP-SGD'):
        optimizer = get_dp_optimizer('SGD', args.lr, args.clip, noise_multiplier, noise_multiplier_list,
                                     args.batch_size,1, net)
        trainer = SAD_DPSGD(args)
        trainer.train(net, train_loader, val_loader, optimizer, criterion, args.R , args.qR , args)

    elif (args.Method == 'Auto-DP-SGD-L'):
        optimizer = get_dp_optimizer('SGD', args.lr, args.clip, noise_multiplier, noise_multiplier_list,
                                     args.batch_size, 1, net)
        trainer = AutoDP(args)

        trainer.train(net, train_loader, val_loader, optimizer, criterion, args)
    elif (args.Method == 'Auto-DP-SGD-S'):
        optimizer = get_dp_optimizer('SGD', args.lr, args.clip, noise_multiplier, noise_multiplier_list,
                                     args.batch_size, 1, net)
        args.qR = 1.0
        trainer = SAD_DPSGD(args)
        trainer.train(net, train_loader, val_loader, optimizer, criterion, args.R, args.qR, args)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

    ## general arguments
    parser.add_argument('--Method', default='SAD-DP-SGD')
    parser.add_argument('--weight_decay', default=0, type=float)

    parser.add_argument('--n_epoch', default=50, type=int, help='total number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='base learning rate (default=0.1)')
    parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')


    ## arguments for learning with differential privacy
    parser.add_argument('--private', default=True, help='enable differential privacy')
    parser.add_argument('--clip', default=20, type=float, help='gradient clipping bound')
    parser.add_argument('--eps', default=3.0, type=float, help='privacy parameter epsilon')
    parser.add_argument('--delta', default=1e-3, type=float, help='desired delta')
    parser.add_argument('--sigma', default=None, type=float, help='desired sigma')
    parser.add_argument('--batch_size', default=128, type=int, help='the size of a batch')

    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])

    parser.add_argument('--R', default=0.8, type=float)
    parser.add_argument('--qR', default=0.7, type=float)
    parser.add_argument('--numOfstep',default=3, type=int)
    args = parser.parse_args()


    DPHAM10000(args)

