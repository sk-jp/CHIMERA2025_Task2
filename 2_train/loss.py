import torch
from torch.nn.modules.loss import _Loss

from torch.nn import CrossEntropyLoss

class Loss(_Loss):
    def __init__(self, conf):
        super(Loss, self).__init__()

        keys = conf.lossfun_keys
        weights = torch.tensor(conf.lossfun_weights)
        
        lossfuns = dict()
        lossfun_weights = dict()
        for idx, key in enumerate(keys):
            if conf[key].params is None:
                lossfun = eval(conf[key].lossfun)()
            else:
                if 'weight' in conf[key].params.keys():
                    conf[key].params['weight'] = torch.tensor(conf[key].params['weight']).type_as(weights)
                lossfun = eval(conf[key].lossfun)(**conf[key].params)
            lossfuns[key] = lossfun
            lossfun_weights[key] = weights[idx]

        self.lossfuns = lossfuns
        self.lossfun_weights = lossfun_weights

    def forward(self, pred, target, **kwargs):
        total_loss = 0
        losses = dict()
        
        for key in self.lossfuns.keys():
            device = pred.device

            if hasattr(self.lossfuns[key], "weight"):
                if self.lossfuns[key].weight is not None:
                    self.lossfuns[key].weight = self.lossfuns[key].weight.to(device)

            loss = self.lossfuns[key](pred, target, **kwargs)
            total_loss += self.lossfun_weights[key].to(device) * loss
            losses[key] = loss
            
        return total_loss, losses
    