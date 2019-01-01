# config file
import torch
import argparse
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import Model

parser = argparse.ArgumentParser(description='SPNN')
parser.add_argument('arch', metavar='ARCH',
                    help='model architecture: lenet53,lenet5,vgg16')
parser.add_argument('dataset', metavar='DATASET',
                    help='dataset name: MNIST,CIFAR10')
parser.add_argument('path_data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('path_model', metavar='PRETRAINED',
                    help='pretrained model file')
parser.add_argument('--path_check', default='checkpoint/z.pk', metavar='CHECK',
                    help='path to checkpoint')
parser.add_argument('--save_format', default='net', metavar='FORMAT',
                    help='save format, net,state_dict')
parser.add_argument('target_acc', metavar='TARGETACC', type=float,
                    help='target acc: vgg16(0.911), lenet5(0.9911), lenet53(0.9801)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr_forward', default=1e-3, type=float,
                    metavar='LRF', help='forward learning rate, default=1e-3')
parser.add_argument('--lr_adv', default=1e-2, type=float,
                    metavar='LRA', help='adv learning rate, default=1e-2')
parser.add_argument('--adv_eps', default=0.1, type=float,
                    metavar='EPS', help='adv bound, default=0.1')
parser.add_argument('--stop_diff', default=1, type=float,
                    metavar='SD', help='adv training stopping if diff < SD, default=1')
parser.add_argument('--thres', default=0.9, type=float,
                    metavar='TH', help='clip elements < adv_eps*thres, default: 0.9')
parser.add_argument('-T', '--epoch', default=30, type=int,
                    metavar='T', help='forward training epoch, default=20')
parser.add_argument('--tb', '--train-batch', default=-1, type=int,
                    metavar='TB', help='forward training batch number, default=-1(full batch)')
args = parser.parse_args()

# ------------ dataset -----------
if args.dataset == 'CIFAR10':
    # use cifar10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    img_size = 32
    n_channel = 3
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=args.path_data, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    testset = torchvision.datasets.CIFAR10(root=args.path_data, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.workers)
    advloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    
elif args.dataset == 'MNIST':
    # use mnist
    img_size = 28
    n_channel = 1
    trainloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=False, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    advloader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(args.path_data, train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size, shuffle=True)
    

# ------------ net arch ----------
net = torch.load(args.path_model)
# print(net)

if args.arch in ['vgg16']:
    features_layerid = [0,3,7,10,14,17,20,24,27,30,34,37,40]
    classifier_layerid = []
elif args.arch == 'lenet5':
    features_layerid = [0,3]
    classifier_layerid = [0]
elif args.arch == 'lenet53':
    features_layerid = []
    classifier_layerid = [0,2]


# build masks
mask_list = []
for i in features_layerid:
    mask_list.append(torch.ones_like(net.features[i].bias))
mask_list.append(torch.ones(net.classifier[0].in_features))
for i in classifier_layerid:
    mask_list.append(torch.ones_like(net.classifier[i].bias))
n_layer = len(mask_list)

# build net with mask
if args.arch == 'lenet53':
    net = Model.FC_Mask(net, mask_list, classifier_layerid)
else:
    net = Model.CONV_Mask(net, mask_list, features_layerid, classifier_layerid)
net = net.cuda()
print(net)


# ------------ others ----------
loss_ce = nn.CrossEntropyLoss()
alpha = 1e-2
ITER_NUM = 10000
LARGE_NUM = 10000
SMALL_NUM = 1e-3

