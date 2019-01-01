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
        loss = -loss_ce(outputs, targets) # + alpha*loss_mee(F.softmax(outputs), args.batch_size) # adv loss + mee loss
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

for layerid in range(n_layer):
    print('---------------- start layer ',layerid,' ---------------')
    wait = 0
    inc = 1
    eps = args.adv_eps
    mask_prev = deepcopy(net.mask[layerid].data)
    ratio = [0, torch.numel(net.mask[layerid].data), torch.numel(net.mask[layerid].data)]
    for epoch in range(LARGE_NUM):
        print('$$$$$$$$$$$$$ epoch ', epoch, ' $$$$$$$$$$$$')
        if wait == 5: break

        ## adv train
        clamper = EpsilonClipper(layerid, eps)
        tailer = MiddleClipper(layerid, eps, args.thres, ratio[1]-inc, 1)
        optimizer_adv = optim.Adam([net.mask[layerid]], lr=args.lr_adv)
        loss_prev = 0
        loss_diff = LARGE_NUM
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer_adv, milestones=[3, 4, 7, 10, 20], gamma=0.5)
        cnt = 0
        while loss_diff > args.stop_diff and cnt < args.epoch:
            # scheduler.step()
            adv_loss = train_adv(epoch)
            loss_diff = abs(loss_prev - adv_loss)
            loss_prev = adv_loss
            cnt += 1
            print('adv train loss: ', adv_loss, ', diff: ',loss_diff)
        net.apply(tailer)
        print('layer ',layerid,' adv train finish, try to retain ', net.mask[layerid].data.nonzero().shape[0])

        if net.mask[layerid].data.nonzero().shape[0] >= mask_prev.nonzero().shape[0]:
            net.mask[layerid].data = deepcopy(mask_prev)
            print('>>>>>>> reverse layer ',layerid, ' since no improvement >>>>>>>')
            eps *= 2
            print('==> this epoch: ',ratio[1], '/', ratio[2], ', inc: ',inc,',wait:',wait,',eps:',eps)
            continue

        ## forward training
        optimizer = optim.Adam(net.net.parameters(), lr=args.lr_forward)
        cnt = 0
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.5)
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
        if test_acc < args.target_acc:
            net.mask[layerid].data = deepcopy(mask_prev)
            print('>>>>>>> reverse layer ',layerid, ' since performance drop >>>>>>>')
            wait += 1
            inc = max(int(inc/2), 1)
            print('==> this epoch: ',ratio[1], '/', ratio[2], ', inc: ',inc,',wait:',wait,',eps:',eps)
            continue

        ratio[1] = net.mask[layerid].data.nonzero().shape[0] 
        ratio[0] = ratio[1]/ratio[2]          
       
        mask_prev = deepcopy(net.mask[layerid].data)
        wait = 0
        inc = min(inc*2, int(ratio[1]/2))
        print('==> this epoch: ',ratio[1], '/', ratio[2], ', inc: ',inc,',wait:',wait,',eps:',eps)
        
#     ## print layer results
#     for i in range(n_layer):
#         print('layer ',i,' : ',ratio[i][0], ' ==> ', ratio[i][1],'/',ratio[i][2],', inc: ',inc[i])
#     print('eps',eps,' wait',wait,' inc',inc)

    ## save model or state_dict
    if args.save_format == 'net': torch.save(net, args.path_check)
    else: torch.save(net.state_dict(), args.path_check)

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
