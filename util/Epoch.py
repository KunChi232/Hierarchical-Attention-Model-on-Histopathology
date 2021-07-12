import torch
import torch.nn.functional as F

import tqdm.tqdm as tqdm
from meter import AverageValueMeter
import sys

class Epoch:
    def __init__(self, model, loss, device = 'cuda', stage = 'Train', optimizer = None, positive_count = 0, negative_count = 0):
        self.model = model
        self.loss = loss
        self.device = device
        self.stage = stage
        self.optimizer = optimizer
        self.to_device()

        self.positive_count = positive_count
        self.negative_count = negative_count
        

    def to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        if(self.optimizer):
            self.optimizer.to(self.device)

    def on_epoch_start(self):
        pass

    def batch_update(self):
        pass

    def run(self, dataloder):
        self.on_epoch_start()

        loss_meter = AverageValueMeter()
        logs = {}

        with tqdm(dataloder, desc = self.stage, file = sys.stdout) as iterator:
            for x, y, patch_names in iterator:
                x = [xi.to(self.device) for xi in x]
                y = y.to(self.device)
                
                loss, pred, weights = self.batch_update(x, y)
                loss = loss.cpu().detach().numpy()
                loss_meter.add(loss)
                logs.update({'loss' : loss})

                iterator.set_postfix_str('Loss:'+str(logs['loss']))

        return logs


class TrainEpoch(Epoch):
    def __init__(self):
        super().__init__()

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        pred, instance_loss, cluster_attention_weight = self.model(x, y)
        bag_loss = F.cross_entropy(pred, y, weight = torch.tensor([1/self.negative_count, 1/self.positive_count]))
        total_loss = bag_loss + instance_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss, pred, cluster_attention_weight
