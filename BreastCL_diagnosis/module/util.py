import torch
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler

from monai.utils import ensure_tuple_rep
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification


from module.models_backup_withbn import mask_vit_small_patch16_224 as mask_vit_small_patch16_224_withbn



def get_multimodal_model(args,fuse_tag, models_tag, data_sizes, num_classes):
    T2_model_tag, DWI_model_tag, DCE_model_tag = models_tag
    T2_data_size, DWI_data_size, DCE_data_size = data_sizes
    
    if fuse_tag == 'mask_vit_withbn':
        model = mask_vit_small_patch16_224_withbn()
    
    return model

def get_optimizer(optimizer_tag, model, lr, weight_decay):
    if optimizer_tag == 'Adam':
        optimizer = torch.optim.Adam(
            # model.parameters(),
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_tag == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
    elif optimizer_tag == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9
        )
    return optimizer

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_lr, min_lr, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = (self.last_epoch + 1) / self.total_steps

        if self.last_epoch < self.warmup_steps:
            lr = self.max_lr * (self.last_epoch + 1) / self.warmup_steps
        else:
            cosine_decay = 0.5 * (1 + torch.cos(torch.pi * (progress - self.warmup_steps / self.total_steps)))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        
        return [lr] * len(self.optimizer.param_groups)



