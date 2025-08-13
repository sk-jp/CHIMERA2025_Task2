from __future__ import print_function, division

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class FeatureMetadataLabelDataset(Dataset):
    """ Dataset for image and label """
    def __init__(self, topdir, csv_file, num_classes):
        """
        Args:
            topdir (str): Path to the top dir.
            csv_file (str): Path to the csv file with annotations.
        """
        super(FeatureMetadataLabelDataset, self).__init__()
        
        self.topdir = topdir
        self.data_list = pd.read_csv(csv_file, header=0)
        self.data_list['meta_data'] = None    # new column for meta data
        self.keys = self.data_list.keys()

        # normalization
        for idx in range(len(self.data_list)):
            meta_data = np.array([], dtype=int)

            # age
            age = self.data_list.loc[idx, 'age']
            if age < 0: # missing value
                age = 100
            elif age < 50:
                age = 50
            elif age > 90:
                age = 90
            age = int((age - 50) // 10)
            meta_data = np.append(meta_data, np.identity(6, dtype=int)[age])

            # sex
            if self.data_list.loc[idx, 'sex'] == "Male":
                sex = 0
            elif self.data_list.loc[idx, 'sex'] == "Female":
                sex = 1
            else:
                sex = 2
            meta_data = np.append(meta_data, np.identity(3, dtype=int)[sex])

            # smoking
            if self.data_list.loc[idx, 'smoking'] == "Yes":
                smoking = 0
            elif self.data_list.loc[idx, 'smoking'] == "No":
                smoking = 1
            else:
                smoking = 2
            meta_data = np.append(meta_data, np.identity(3, dtype=int)[smoking])

            # tumor
            if self.data_list.loc[idx, 'tumor'] == "Primary":
                tumor = 0
            elif self.data_list.loc[idx, 'tumor'] == "Recurrence":
                tumor = 1
            meta_data = np.append(meta_data, np.identity(2, dtype=int)[tumor])

            # stage
            if self.data_list.loc[idx, 'stage'] == "TaHG":
                stage = 0
            elif self.data_list.loc[idx, 'stage'] == "T1HG":
                stage = 1
            elif self.data_list.loc[idx, 'stage'] == "T2HG":
                stage = 2
            meta_data = np.append(meta_data, np.identity(3, dtype=int)[stage])

            # substage
            if self.data_list.loc[idx, 'substage'] == "T1m":
                substage = 0
            elif self.data_list.loc[idx, 'substage'] == "T1e":
                substage = 1
            else:
                substage = 2
            meta_data = np.append(meta_data, np.identity(3, dtype=int)[substage])

            # grade
            if self.data_list.loc[idx, 'grade'] == "G2":
                grade = 0
            elif self.data_list.loc[idx, 'grade'] == "G3":
                grade = 1
            meta_data = np.append(meta_data, np.identity(2, dtype=int)[grade])

            # reTUR
            if self.data_list.loc[idx, 'reTUR'] == "Yes":
                retur = 0
            elif self.data_list.loc[idx, 'reTUR'] == "No":
                retur = 1
            meta_data = np.append(meta_data, np.identity(2, dtype=int)[retur])

            # LVI
            if self.data_list.loc[idx, 'LVI'] == "Yes":
                lvi = 0
            elif self.data_list.loc[idx, 'LVI'] == "No":
                lvi = 1
            else:
                lvi = 2
            meta_data = np.append(meta_data, np.identity(3, dtype=int)[lvi])

            # variant
            if self.data_list.loc[idx, 'variant'] == "UCC":
                variant = 0
            elif self.data_list.loc[idx, 'variant'] == "UCC + Variant":
                variant = 1
            meta_data = np.append(meta_data, np.identity(2, dtype=int)[variant])

            # EORTC
            if self.data_list.loc[idx, 'EORTC'] == "High risk":
                eortc = 0
            elif self.data_list.loc[idx, 'EORTC'] == "Highest risk":
                eortc = 1
            else:
                eortc = 2
            meta_data = np.append(meta_data, np.identity(3, dtype=int)[eortc])

            # no_instillations
            no_inst = self.data_list.loc[idx, 'no_instillations']            
            if no_inst > 50:
                no_inst = 50
            elif no_inst < 0:
                no_inst = 60
            no_inst = int(no_inst // 10)
            meta_data = np.append(meta_data, np.identity(7, dtype=int)[no_inst])

            meta_data = ''.join(map(str, meta_data))
            self.data_list.loc[idx, 'meta_data'] = meta_data

            # BRS (reference)
            if num_classes == 2:
                if self.data_list.loc[idx, 'BRS'] == "BRS1" or \
                    self.data_list.loc[idx, 'BRS'] == "BRS2":
                    self.data_list.loc[idx, 'BRS'] = 0
                elif self.data_list.loc[idx, 'BRS'] == "BRS3":
                    self.data_list.loc[idx, 'BRS'] = 1
            elif num_classes == 3:
                if self.data_list.loc[idx, 'BRS'] == "BRS1":
                    self.data_list.loc[idx, 'BRS'] = 0
                elif self.data_list.loc[idx, 'BRS'] == "BRS2":
                    self.data_list.loc[idx, 'BRS'] = 1
                elif self.data_list.loc[idx, 'BRS'] == "BRS3":
                    self.data_list.loc[idx, 'BRS'] = 2

        self.meta_data_keys = ["age", "sex", "smoking", "tumor", "stage", 
                               "substage", "grade", "reTUR", "LVI", "variant", 
                               "EORTC", "no_instillations"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # feature vector
        case_name = self.data_list.iloc[idx, 0]  
        feature_file = f"{self.topdir}/{case_name}_HE.pt"
        feature = torch.load(feature_file)

        # meta (clinical) data
        meta_data = self.data_list.iloc[idx]['meta_data']
        meta_data = np.array([int(c) for c in meta_data])
        meta_data = torch.tensor(meta_data, dtype=torch.float32)

        # label
        label = self.data_list.iloc[idx]['BRS']
        
        # return values
        ret = {}
        ret['feature'] = feature
        ret['meta_data'] = meta_data
        ret['label'] = label
        ret['case_name'] = case_name

        return ret

