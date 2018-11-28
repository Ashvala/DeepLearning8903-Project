import os
import numpy as np
import torch
import torch.utils.data
import json
import librosa
from torch.autograd import Variable

torch.manual_seed(0)
np.random.seed(0)

class nsynthDataset(torch.utils.data.Dataset):
    """ 
    Dataset loader
    """
    def __init__(self, datapath='/storage/', mode='train', length=None):
        super(nsynthDataset, self).__init__()
        self.datapath = datapath
        self.mode = mode
        audio_path = "{}/nsynth-{}/audio".format(self.datapath, self.mode)
        examples_path = "{}/nsynth-{}/examples.json".format(self.datapath, self.mode)
        with open(examples_path) as f:
            self.examples = json.load(f)

        self.data_files = os.listdir(audio_path)
            
        if length != None:
            self.data_files = self.data_files[0:length]

        if mode=='valid':
            self.data_files = self.data_files[0:len(self.data_files)//4]
        
    def __len__(self):
        return len(self.data_files)
        
    def label_vectorize(self, i):
        # [0-127]
        vector_arr = np.zeros(128)
        vector_arr[i] = 1
        return vector_arr
    
    def __getitem__(self, idx):
        split_str = self.data_files[idx].split(".")
        key = split_str[0]
        pitch = self.examples[key]['pitch']
        vel_map = {
            25: [1, 0, 0, 0, 0],
            50: [0, 1, 0, 0, 0],
            75: [0, 0, 1, 0, 0], 
            100: [0, 0, 0, 1, 0], 
            127: [0, 0, 0, 0, 1]
        }
        velocity = self.examples[key]['velocity']
        data_path = '{}/nsynth-{}/audio/{}'.format(self.datapath, self.mode, self.data_files[idx])
        sig, sr = librosa.load(data_path,  sr=16000)
        return torch.FloatTensor(sig), torch.LongTensor([pitch, velocity])

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def evaluate(data_loader, model, criterion, cuda):
    """ Evaluate the network! """
    model.eval()
    with torch.no_grad():
        loss = 0
        correct = 0
        n_examples = 0
        beta = 0.6
        loader = data_loader
        accuracy_list = []
        for batch_i, (data,t_pitch) in enumerate(loader):
            if model.cuda:
                data, t_pitch = data.cuda(), t_pitch.cuda()
            pitch = model(data)
            pitch_loss = criterion(pitch, t_pitch[:,0])
            vel_loss = 0
            loss = pitch_loss + beta * vel_loss
            pred = pitch.data.max(1, keepdim=True)[1]
            accuracy_list.append(accuracy(pitch, t_pitch[:, 0]))
            correct += pred.eq(t_pitch[:,0].data.view_as(pred)).sum()
            n_examples += pred.size(0)
        loss /= n_examples
        acc = float(correct) / n_examples
        return loss, acc


def save(model, epoch, loss, optimizer, path):
    torch.save(model, path)


def load(path):
    model = torch.load(path)
    state_dict = model.state_dict()
    return state_dict
