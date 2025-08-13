import torch
import lightning.pytorch as pl

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, BinaryF1Score

from get_optimizer import get_optimizer
from loss import Loss

class LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(LightningModule, self).__init__()
        self.cfg = cfg
        
        # model
        if cfg.Model.arch == 'mlp':
            from torchvision.ops import MLP
            if 'norm_layer' in cfg.Model.params.keys():
                cfg.Model.params['norm_layer'] = eval(cfg.Model.params['norm_layer'])
            if 'activation_layer' in cfg.Model.params.keys():
                cfg.Model.params['activation_layer'] = eval(cfg.Model.params['activation_layer'])
            self.model = MLP(**cfg.Model.params)
        else:
            raise ValueError(f'{cfg.Model.arch} is not supported.')

        # output buffers
        self.training_step_outputs = []        
        self.validation_step_outputs = []

        # metrics
        num_classes = cfg.Data.dataset.num_classes
        metrics_fun = MetricCollection([
            MulticlassAccuracy(num_classes=num_classes, average='micro'),
            MulticlassPrecision(num_classes=num_classes, average='macro'),
            MulticlassRecall(num_classes=num_classes, average='macro'),
            MulticlassF1Score(num_classes=num_classes, average='macro')
        ])
        self.train_metrics_list = ['MulticlassAccuracy', 'MulticlassPrecision', 'MulticlassRecall', 'MulticlassF1Score']
        self.valid_metrics_list = ['MulticlassAccuracy', 'MulticlassPrecision', 'MulticlassRecall', 'MulticlassF1Score']

        self.train_metrics_fun = metrics_fun.clone(prefix='train_')
        self.valid_metrics_fun = metrics_fun.clone(prefix='valid_')

        # flag to check the validation is performed or not at the end of epoch
        self.did_validation = False

    def setup(self, stage=None):
        if stage == "fit":
            self.lossfun = Loss(self.cfg.Loss)
            self.lossfun_valid = Loss(self.cfg.Loss)
        elif stage == "validate":
            self.lossfun_valid = Loss(self.cfg.Loss)

    def forward(self, x, **kwargs):
        y = self.model(x, **kwargs)
        return y

    def on_train_epoch_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()
        self.did_validation = False

    def training_step(self, batch, batch_idx):
        feature = batch["feature"]      # [b,768]
        meta_data = batch["meta_data"]  # [b,12]
        label = batch["label"]     # [b,1]

        # forward
        if self.cfg.Model.arch == 'mlp':
            x = torch.cat((feature, meta_data), dim=1)  # [b,780]   

        y = self.forward(x)

        # loss
        loss, _ = self.lossfun(y, label)
                               
        output = {"loss": loss.item()}
        self.training_step_outputs.append(output)

        # metrics
        for metric in self.train_metrics_list:
            self.train_metrics_fun[metric].update(preds=y, target=label)

        return loss

    def on_train_epoch_end(self):
        # print the results
        outputs_gather = self.all_gather(self.training_step_outputs)

        # metrics (each metric is synced and reduced across each process)
        train_metrics = self.train_metrics_fun.compute()
        self.train_metrics_fun.reset()

        if self.trainer.is_global_zero:
            epoch = int(self.current_epoch)

            # loss
            train_loss = torch.stack([o['loss']
                                      for o in outputs_gather]).mean().detach()

            # log
            d = dict()
            d['epoch'] = epoch
            d['train_loss'] = train_loss
            d.update(train_metrics)

            print('\n Mean:')
            s = f'  Train:\n'
            s += f'    loss: {train_loss.item():.3f}'
            s += '\n'
            s += '  '
            for metric in self.train_metrics_list:
                s += f'  {metric.replace("Multiclass", "")}: {train_metrics[f"train_{metric}"].cpu().numpy():.3f}'
            print(s)

            if self.did_validation:
                s = '  Valid:\n'
                s += f'    loss: {self.valid_loss:.3f}'
                s += '\n'
                s += '  '
                for metric in self.valid_metrics_list:
                    s += f'  {metric.replace("Multiclass", "")}: {self.valid_metrics[f"valid_{metric}"].cpu().numpy():.3f}'
                print(s)

            self.log_dict(d, prog_bar=False, rank_zero_only=True)

        self.training_step_outputs.clear()

    def on_validation_epoch_start(self):
        # clear GPU cache before the validation
        torch.cuda.empty_cache()
        pl.utilities.memory.garbage_collection_cuda()        
     
    def validation_step(self, batch, batch_idx):
        feature = batch["feature"]      # [b,768]
        meta_data = batch["meta_data"]  # [b,12]
        label = batch["label"]     # [b,1]

        # forward
        if self.cfg.Model.arch == 'mlp':
            x = torch.cat((feature, meta_data), dim=1)  # [b,807]   

        y = self.forward(x)

        # loss
        loss, _ = self.lossfun_valid(y, label)

        # metrics
        for metric in self.valid_metrics_list:
            self.valid_metrics_fun[metric].update(preds=y, target=label)

        # output buffer
        output = {"loss": loss}
        self.validation_step_outputs.append(output)

        return
   
    def on_validation_epoch_end(self):
        # all gather
        outputs_gather = self.all_gather(self.validation_step_outputs)

        # loss
        valid_loss = torch.stack([o['loss'] for o in outputs_gather]).mean().item()
        self.valid_loss = valid_loss

        # metrics (each metric is synced and reduced across each process)
        self.valid_metrics = self.valid_metrics_fun.compute()
        self.valid_metrics_fun.reset()

        # log
        epoch = int(self.current_epoch)
        d = dict()
        d['epoch'] = epoch
        d['valid_loss'] = valid_loss
        for key in self.valid_metrics.keys():
            d[f'valid_{key}'] = self.valid_metrics[key]

        self.log_dict(d, prog_bar=False, rank_zero_only=True)

        # free up the memory
        self.validation_step_outputs.clear()

        # setup flag
        self.did_validation = True

    def on_test_start(self):
        pass

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def on_test_end(self):
        pass

    def on_predict_start(self):
        pass

    def predict_step(self, batch, batch_idx):
        pass

    def on_predict_end(self):
        pass

    def configure_optimizers(self):
        conf_optim = self.cfg.Optimizer

        if hasattr(conf_optim.optimizer, 'params'):
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters(),
                                      **conf_optim.optimizer.params)
        else:
            optimizer_cls, scheduler_cls = get_optimizer(conf_optim)
            optimizer = optimizer_cls(self.parameters())

        if scheduler_cls is None:
            return {"optimizer": optimizer}
        else:
            scheduler = scheduler_cls(
                optimizer, **conf_optim.lr_scheduler.params)
            
            return {"optimizer": optimizer,
                    "lr_scheduler": scheduler}
        
    def get_progress_bar_dict(self):
        items = dict()

        return items

