import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
import gc
import torch.nn.functional as F
import datetime
import torch.optim as optim
from utils import evaluate, save, nsynthDataset, load
from Alvin import *
import scipy
import subprocess
import librosa
from tensorboardX import SummaryWriter
writer = SummaryWriter()

# Training settings
parser = argparse.ArgumentParser(description='HW 2: Music/Speech CNN')

# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=1e-3,
                    help='learning rate')
parser.add_argument('--momentum', type=float, metavar='M',
                     help='SGD momentum', default=0.95)
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=25,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=50,
                    help='number of epochs to train')
parser.add_argument('--save-dir', default='models/')
parser.add_argument('--file-name', default='alvin')
parser.add_argument('--load_from_point', default=None)
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(0)
np.random.seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)


print("preparing dataset")
train_set = nsynthDataset(datapath='./data/nsynth/', mode='train')
val_set = nsynthDataset(datapath='./data/nsynth/', mode='valid')
test_set = nsynthDataset(datapath='./data/nsynth/', mode='test')

print("creating dataloaders")
train_loader = torch.utils.data.DataLoader(train_set, batch_size = args.batch_size, shuffle=True, num_workers=4)
val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32, num_workers=4)



model = alvin_big(num_classes=128)

if args.load_from_point == None:
    print(model)
else:
    state_dict = load(args.load_from_point)
    model.load_state_dict(state_dict)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss().cuda()

start_time = datetime.datetime.now()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

def train(epoch):
    print("In the train step")
    mean_training_loss = 0.0
    model.train()
    beta = 0.01
    for i, (data, t_pitch) in enumerate(train_loader):
        # add cuda flags here
        if args.cuda:
            data, t_pitch = data.cuda(), t_pitch.cuda()
        pitch = model(data)
        
        pitch_loss = criterion(pitch, t_pitch[:,0])
        vel_loss = 0
        total_loss = pitch_loss + (beta * vel_loss)
        mean_training_loss += total_loss
        elapsed = datetime.datetime.now() - start_time

        optimizer.zero_grad()
        pitch_loss.backward()        
        optimizer.step()
        writer.add_scalar('Train/Loss', pitch_loss, (epoch*len(train_loader)+i))
        if i%100 == 0: 
            print("{} [{}][{}/{}] {}".format(elapsed, epoch, i, len(train_loader), total_loss.item()))        

    mean_training_loss = mean_training_loss.item()/len(train_loader)
    print('Training Epoch: [{}][{}/{}]\t'
            'Training Loss: {}'.format(
            (epoch), (i), len(train_loader) - 1, mean_training_loss))

    del pitch, mean_training_loss
    torch.cuda.empty_cache()


def run(epochs, train_loader, val_loader, test_loader):
    for i in range(epochs):
        train(i)
        print ("training at: ", i)
        val_loss, val_acc = evaluate(val_loader, model, criterion, args.cuda)
        writer.add_scalar('Validation/Loss', val_loss, i)
        writer.add_scalar('Validation/Accuracy', val_acc, i)        
        print('Validation Loss: {:.6f} \t'
                'Validation Acc.: {:.6f}'.format(
                val_loss, val_acc))        
        save_file = args.file_name + "_epoch_" +  str(i) + ".pth"
        print("saving: ", save_file)
        save(model, i, val_loss, optimizer, args.save_dir + save_file)
    test(model, test_loader)
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def test(model, test_loader):
    test_loss, test_acc = evaluate(test_loader, model, criterion, args.cuda)
    print('Test Loss: {:.6f} \t'
            'Test Acc.: {:.6f}'.format(
            test_loss, test_acc))


if __name__ == "__main__":
    run(args.epochs, train_loader, val_loader, test_loader)



