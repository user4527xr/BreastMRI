import torch
from torchvision import transforms as T
from dataset.mri import MultiModalDataset, MultiModalDataset_Test
from dataset.augmentation import Inserter, Flipper, MultimodalInserter, MultimodalFlipper, MultimodalRotater, MultimodalResizer,\
                              MultimodalSixFlipper, MultimodalNineInserter

transforms = {
    'DCE':{
        "train": T.Compose([Inserter(size=(336, 224, 128)), Flipper()]),
        "val": T.Compose([Inserter(size=(336, 224, 128), rand=False)]),
        "test": T.Compose([Inserter(size=(336, 224, 128), rand=False)])
        },
    'T2':{
        "train": T.Compose([Inserter(size=(384, 384, 48)), Flipper()]),
        "val": T.Compose([Inserter(size=(384, 384, 48), rand=False)]),
        "test": T.Compose([Inserter(size=(384, 384, 48), rand=False)])
        },
    'DWI':{
        "train": T.Compose([Inserter(size=(256, 256, 32)), Flipper()]),
        "val": T.Compose([Inserter(size=(256, 256, 32), rand=False)]),
        "test": T.Compose([Inserter(size=(256, 256, 32), rand=False)])
        },
    'ADC':{
        "train": T.Compose([Inserter(size=(256, 128, 32)), Flipper()]),
        "val": T.Compose([Inserter(size=(256, 128, 32), rand=False)]),
        "test": T.Compose([Inserter(size=(256, 128, 32), rand=False)])
        },
    'Multi':{
        "train": T.Compose([MultimodalInserter(dce_size=(336, 224, 128),
                                               dwi_size=(258, 128, 32),
                                               t2_size=(336, 224, 48))]),
        "val": T.Compose([MultimodalInserter(dce_size=(336, 224, 128),
                                               dwi_size=(258, 128, 32),
                                               t2_size=(336, 224, 48), rand=False)]),
        "test": T.Compose([MultimodalInserter(dce_size=(336, 224, 128),
                                               dwi_size=(258, 128, 32),
                                               t2_size=(336, 224, 48), rand=False)]),#
        
        },
}


def get_dataset(modal, dataset_split, transform_split, root='', task='report', fold=0, jud='YN'):
    root = root
    root = '/home/csexrjiang/20260418_BreastRG/datacsv'
    
    if modal == 'Multi':
        dataset = MultiModalDataset(
                  split = dataset_split,
                  root = root,
                  transform=transforms[modal][transform_split],
                  transform_dce = transforms['DCE'][transform_split],
                  task=task,
                  fold=fold)
        
    elif modal == 'Multi_Test':
        dataset = MultiModalDataset_Test(
                  split=dataset_split,
                  jud = jud,
                  root=root,
                  transform=transforms['Multi'][transform_split],
                  transform_dce = transforms['DCE'][transform_split],
                  task=task,
                  fold=fold)       
    else:
        raise KeyError(f"This dataload function is not yet implemented.")
    
    return dataset


