import torch

class MiddleClipper(object):
    def __init__(self, _layerid, _epsilon, _thres, _k, _center):
        self.epsilon = _epsilon
        self.layerid = _layerid
        self.thres = _thres
        self.k = _k
        self.center = _center
    def __call__(self, module):
        if hasattr(module, 'mask'):
            w = module.mask[self.layerid].data 
            w -= self.center # re-center to 1
            if any(e == False for e in (torch.abs(w) < self.epsilon)):
                w[torch.abs(w) > self.epsilon*self.thres] = 1
            else: # set largest to 1
                print('************ all values are small in this layer **********')#' Frob(w)/|w| = ', torch.norm(w, p=2))
                w[torch.topk(torch.abs(w),max(self.k,1))[1]] = 1
            w[w != 1] = 0 # make remaining w to be 0

class EpsilonClipper(object):
    def __init__(self, _layerid, _epsilon):
        self.epsilon = _epsilon
        self.layerid = _layerid
    def __call__(self, module):
        if hasattr(module, 'mask'):
            w = module.mask[self.layerid].data
            w.clamp_(1-self.epsilon, 1+self.epsilon)

def accuracy(output, target, topk=(1,)): 
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object): 
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
