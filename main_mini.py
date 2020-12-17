from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
import glob
from sklearn.model_selection import KFold
from utils.MPN_dataloader import MPNBags
from utils.model import Attention, GatedAttention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MPN Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')

parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
dataset_path = './bag_data'
# load datapath from path
c1 = glob.glob(dataset_path+'/0/*M*')
c2 = glob.glob(dataset_path+'/1/*M*')
c3 = glob.glob(dataset_path+'/2/*M*')
c4 = glob.glob(dataset_path+'/3/*M*')

c1_num = len(c1)
c2_num = len(c2)
c3_num = len(c3)
c4_num = len(c4)
all_path = c1 + c2 + c3 + c4

kf = KFold(n_splits=4, shuffle=True, random_state=1)
datasets = []
for train_idx, test_idx in kf.split(all_path):
    dataset = {}
    dataset['train'] = [all_path[ibag] for ibag in train_idx]
    dataset['test'] = [all_path[ibag] for ibag in test_idx]
    datasets.append(dataset)
    
train_loader = data_utils.DataLoader(MPNBags(datasets_all=dataset,seed=1,train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

# test_loader = data_utils.DataLoader(MPNBags(datasets_all=dataset,seed=1,train=False),
#                                     batch_size=1,
#                                     shuffle=False,
#                                     **loader_kwargs)
print('Load Train and Test Set: Done!!!')
print('Init Model')
if args.model=='attention':
    model = Attention()
    print(model)
elif args.model=='gated_attention':
    model = GatedAttention()
    print(model)
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data.item()
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    for batch_idx, (data, label) in enumerate(test_loader):
        bag_label = label[0]
        instance_labels = label[1]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)
        loss, attention_weights = model.calculate_objective(data, bag_label)
        test_loss += loss.data[0]
        error, predicted_label = model.calculate_classification_error(data, bag_label)
        test_error += error

        if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
            bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
            instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                 np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

            print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                  'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
