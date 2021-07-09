import tqdm.tqdm as tqdm

from meter import AverageValueMeter

import sys

class Epoch:
    def __init__(self, model, loss, device = 'cuda', stage = 'Train'):
        self.model = model
        self.loss = loss
        self.device = device
        self.stage = stage

        self.to_device()

    def to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

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
                
                loss = self.batch_update()
                loss = loss.cpu().detach().numpy()
                loss_meter.add(loss)
                logs.update({'loss' : loss})

                iterator.set_postfix_str('Loss:'+str(logs['loss']))

        return logs
