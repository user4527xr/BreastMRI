import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer

def get_csv_file(split):
    if split == 'train':
        csv_file = 'example_train.csv'
        print('Training with: ', csv_file)
    elif split == 'val':
        csv_file = 'example_val.csv'
        print('Validation: ', csv_file)
    elif split == 'test':       
        csv_file = 'example_test.csv'       
        print('Testing: ', csv_file)
    return csv_file

class MultiModalDataset(Dataset):
    def __init__(self, split, root,  transform=None, transform_dce=None, task='report', fold=0):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(MultiModalDataset, self).__init__()
        
        csv_file = get_csv_file(split)
            
        self.df_root = pd.read_csv(os.path.join(root, csv_file))
          
        self.t2_path_root =  [i.replace('/jhcnas4/SZRM_MRI/', '/scratch/medimgfmod/Breast_MRI/DS2/SZRM_batch_3/') for i in self.df_root['T2']]
        self.dwi_path_root = [i.replace('/jhcnas4/SZRM_MRI/', '/scratch/medimgfmod/Breast_MRI/DS2/SZRM_batch_3/') for i in self.df_root['DWI']]
        self.sub_path_root = [i.replace('/jhcnas4/SZRM_MRI/', '/scratch/medimgfmod/Breast_MRI/DS2/SZRM_batch_3/') for i in self.df_root['SUB_concate']]
        self.report_root = [i for i in self.df_root['完整报告']]
        self.index_root = [i for i in self.df_root['Subject']]            
        self.t2_path =  self.t2_path_root 
        self.dwi_path = self.dwi_path_root 
        self.sub_path = self.sub_path_root

        self.report = self.report_root 
        self.index = self.index_root
     
        self.labels_root = [i for i in self.df_root['malignant']]
        self.labels = self.labels_root  
        self.labels = torch.LongTensor(self.labels)
        
        self.transform = transform
        self.transform_dce = transform_dce

    def __getitem__(self, index):
        """
    Args:
        index: the index of item
    Returns:
        image and its labels
        """
        
        t2_pth = self.t2_path[index]
        dwi_pth = self.dwi_path[index]
        sub_pth = self.sub_path[index]

        
        t2 = np.load(t2_pth)[np.newaxis, :]  
        dwi = np.load(dwi_pth)[np.newaxis, :]  
        sub = np.load(sub_pth)  
        
        report = self.report[index]
        report = report.strip()
            
        report = ' '.join(report.split())

        
        labels = self.labels[index]
        i = self.index[index]
               
        if self.transform is not None:
                data = self.transform({'dce':sub, 'dwi':dwi, 't2':t2})
                sub, dwi, t2 = data['dce'], data['dwi'], data['t2'] 
         
        
        to_return = {'id': i}
        to_return['sub'] = sub
        to_return['t2'] = t2
        to_return['dwi'] = dwi
        to_return['report'] = report
        to_return['labels'] = labels
       

        return to_return

    def __len__(self):
        return len(self.sub_path)



class MultiModalDataset_Test(Dataset):
    def __init__(self, split, root, jud, transform=None, transform_dce=None, task='report', fold=0):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(MultiModalDataset_Test, self).__init__()
        
        csv_file_test = get_csv_file(split)
       
        self.df = pd.read_csv(os.path.join(root, csv_file_test))
        
        if jud=='YN':
            self.t2_path = [i.replace('/ssd3/data/YN_BC_MRI', '/jhcnas4/YN_BC_MRI/YN_BC_MRI') for i in self.df['T2']]
            self.dwi_path = [i.replace('/ssd3/data/YN_BC_MRI', '/jhcnas4/YN_BC_MRI/YN_BC_MRI') for i in self.df['DWI']]
            self.sub_path = [i.replace('/ssd3/data/YN_BC_MRI', '/jhcnas4/YN_BC_MRI/YN_BC_MRI') for i in self.df['SUB_concate']]
            self.student = [i.replace('/ssd3/data/YN_BC_MRI', '/jhcnas5/xinrui/YN_BC_MRI/YN_BC_MRI') for i in self.df['pre']]

            self.t2_path = self.t2_path 
            self.dwi_path = self.dwi_path 
            self.sub_path = self.sub_path 
            self.student = self.student 
        elif jud=='DS1':
            self.t2_path =  '/home/csexrjiang/20260418_BreastRG/datacsv/example_data' +'/'+self.df['T2']
            self.dwi_path = '/home/csexrjiang/20260418_BreastRG/datacsv/example_data' +'/'+self.df['DWI']
            self.sub_path = '/home/csexrjiang/20260418_BreastRG/datacsv/example_data' +'/'+self.df['SUB_concate']
            # self.t2_path =  self.df['T2']
            # self.dwi_path = self.df['DWI']
            # self.sub_path = self.df['SUB_concate']
           
            self.report = self.df['完整报告']
            self.index = [i for i in self.df['Subject']]

 
    
            self.labels = [i for i in self.df['malignant']]

            self.labels = self.labels 
            self.labels = torch.LongTensor(self.labels)

        
        self.transform = transform
        self.transform_dce = transform_dce

    def __getitem__(self, index):
        """
    Args:
        index: the index of item
    Returns:
        image and its labels
        """
        
        t2_pth = self.t2_path[index]
        dwi_pth = self.dwi_path[index]
        sub_pth = self.sub_path[index]

         
        t2 = np.load(t2_pth)[np.newaxis, :]  
        dwi = np.load(dwi_pth)[np.newaxis, :]  
        sub = np.load(sub_pth)  
    
        
        report = self.report[index]
        report = report.strip()
            
        report = ' '.join(report.split())
    

        labels = self.labels[index]
        i = self.index[index]

        if self.transform is not None:
            data = self.transform({'dce':sub, 'dwi':dwi, 't2':t2})
            sub, dwi, t2 = data['dce'], data['dwi'], data['t2']    

        to_return = {'id': i}
        to_return['sub'] = sub
        to_return['t2'] = t2
        to_return['dwi'] = dwi
        to_return['report'] = report
        to_return['labels'] = labels
        return to_return

    def __len__(self):
        return len(self.sub_path)