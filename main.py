from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import sys
from copy import deepcopy
import Model
from Util import *

# get args and run configs
from config import *

def train_forward(epoch):
    net.train()
    train_loss = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if args.tb != -1 and  batch_idx >= args.tb: break
        inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_ce(outputs, targets)
        loss.backward(retain_graph=True)
        optimizer.step()
        train_loss += loss.item()
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))
    print('forward train acc: top1 -> ',top1.avg, '; top5 -> ',top5.avg, ' and loss: ', train_loss)
    return top1.avg/100, train_loss

def train_adv(epoch):
    net.train()
    train_loss = 0
    for batch_idx, (inputs, targets) in enumerate(advloader):
        inputs, targets = inputs.cuda(async=True), targets.cuda(async=True)
        optimizer_adv.zero_grad()
        outputs = net(inputs)
        loss = -loss_ce(outputs, targets) # adv loss
        loss.backward(retain_graph=True)
        optimizer_adv.step()
        train_loss += loss.item()
        net.apply(clamper)
    return train_loss
        
def test(epoch):
    net.eval()
    test_loss = 0
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_ce(outputs, targets)
            test_loss += loss.item()
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))
    print('test acc: top1 -> ',top1.avg, '; top5 -> ',top5.avg, ' and loss: ', test_loss)
    return top1.avg/100, test_loss

# adv_save = []
mask_prev = []
ratio = []
for i in range(n_layer):
    mask_prev.append(deepcopy(net.mask[i].data))
    # adv_save.append(deepcopy(net.mask[i].data))
    ratio.append([0, torch.numel(net.mask[i].data), torch.numel(net.mask[i].data)])
skip = [0]*n_layer
wait = [0]*n_layer
inc = [1]*n_layer
tol = 0

for epoch in range(LARGE_NUM):
    print('$$$$$$$$$$$$$ epoch ', epoch, ' $$$$$$$$$$$$')
    if all(e > 0 for e in wait): tol += 1
    if tol == 3: break
    for layerid in range(n_layer):
        print('---------------- start layer ',layerid,' ---------------')
        if skip[layerid] == 1: 
            print('### skip layer ', layerid, 'for no improvement ###')
            continue
        if wait[layerid] > 0:
            print('### skip layer ', layerid,'wait: ',wait[layerid], ' ###')
            wait[layerid] -= 1
            continue

        ## adv train
        clamper = EpsilonClipper(layerid, args.adv_eps)
        tailer = MiddleClipper(layerid, args.adv_eps, args.thres, ratio[layerid][1]-inc[layerid], 1)
        optimizer_adv = optim.Adam([net.mask[layerid]], lr=args.lr_adv)
        loss_prev = 0
        loss_diff = LARGE_NUM
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_adv, milestones=[3, 4, 7, 10, 20], gamma=0.5)
        cnt = 0
        # net.mask[layerid].data = deepcopy(adv_save[layerid])
        while loss_diff > args.stop_diff and cnt < args.epoch:
            # scheduler.step()
            adv_loss = train_adv(epoch)
            loss_diff = abs(loss_prev - adv_loss)
            loss_prev = adv_loss
            cnt += 1
            print('adv train loss: ', adv_loss, ', diff: ',loss_diff)
        # adv_save[layerid] = deepcopy(net.mask[layerid].data)
        net.apply(tailer)
        print('layer ',layerid,' adv train finish, try to retain ', net.mask[layerid].data.nonzero().shape[0])

        if net.mask[layerid].data.nonzero().shape[0] >= mask_prev[layerid].nonzero().shape[0]:
            net.mask[layerid].data = deepcopy(mask_prev[layerid])
            print('>>>>>>> reverse layer ',layerid, ' since no improvement >>>>>>>')
            skip[layerid] = 1
            continue

        ## forward training
        optimizer = optim.Adam(net.net.parameters(), lr=args.lr_forward)
        cnt = 0
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
        test_acc, _ = test(epoch)
        loss_prev = LARGE_NUM
        loss_diff = LARGE_NUM
        while(test_acc < args.target_acc and cnt < args.epoch and loss_diff > 1e-3): 
            scheduler.step()
            forward_acc, train_loss = train_forward(epoch)
            loss_diff = abs(loss_prev - train_loss)
            loss_prev = train_loss
            test_acc, _ = test(epoch)
            cnt += 1

        ## check if need reverse
        if test_acc <= args.target_acc:
            net.mask[layerid].data = deepcopy(mask_prev[layerid])
            print('>>>>>>> reverse layer ',layerid, ' since performance drop >>>>>>>')
            print('==> this epoch: ',ratio[layerid][1], '/', ratio[layerid][2],', inc: ', inc[layerid])
            wait[layerid] += 1
            inc[layerid] = max(int(inc[layerid]/2), 1)
            skip[layerid] = 1
            continue

        ratio[layerid][1] = net.mask[layerid].data.nonzero().shape[0] 
        ratio[layerid][0] = ratio[layerid][1]/ratio[layerid][2]          
       
        mask_prev[layerid] = deepcopy(net.mask[layerid].data)
        print('==> this epoch: ',ratio[layerid][1], '/', ratio[layerid][2])
        inc[layerid] *= 2
        print(inc)
        
    ## print layer results
    for i in range(n_layer):
        print('layer ',i,' : ',ratio[i][0], ' ==> ', ratio[i][1],'/',ratio[i][2],', inc: ',inc[layerid])

    ## save model or state_dict
    if args.save_format == 'net': torch.save(net, args.path_check)
    else: torch.save(net.state_dict(), args.path_check)

    if all(e>0 for e in skip):
        skip = [0]*n_layer
        args.adv_eps += 0.1
        print('############# increase adv_eps to ',args.adv_eps, ' #############')
    if args.adv_eps == 10: break
    
#post train
print("-------------- post train ------------")
optimizer = optim.Adam(net.net.parameters(), lr=args.lr_forward/10)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40], gamma=0.5)
for epoch in range(50):
    scheduler.step()
    train_forward(epoch)
    test(epoch)
print("------------- sparsity -----------")
for i in range(n_layer):
    print('layer ',i,' : ',ratio[i][0], ' ==> ', ratio[i][1],'/',ratio[i][2])
 
