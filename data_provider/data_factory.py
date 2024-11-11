# TODO: In this file, define the methods to generate raw data from the file and convert them into 
# 1. dataloader in pytorch
# 2. data in form of numpy.ndarray to be used in sklearn 
# 3. other methods necessary

from data_provider.data_loader import EHR_Dataset, Image_Dataset, Note_Dataset, Multi_Dataset, EHR_DATASET
import numpy as np
from torch.utils.data import DataLoader
from sklearn.utils import shuffle
import pandas as pd
import os
import joblib 

dataset_dict = {"EHR": EHR_Dataset,
                "Image": Image_Dataset, 
                "Note": Note_Dataset,
                "Multi": Multi_Dataset}


def ehr_provider(args, flag):
    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    else:
        if flag == "vali":
            flag = "valid"
        shuffle_flag = True
        drop_last = False
        freq = args.freq
        batch_size = args.batch_size

    sub_df = pd.read_csv(os.path.join(args.root_path, f'{flag}.csv'))
    raw_y = sub_df.drop_duplicates(subset=['id'], keep="first")[['readmitted_within_30days', 'id']]
    instances = np.unique(sub_df["id"].values)
    feature_dict = joblib.load(os.path.join(args.root_path, args.ehr_path))['feat_dict']
    instance_data = []
    for sub in instances:
        data_ = dataset_dict['EHR'](sub, feature_dict[sub], int(raw_y[raw_y['id'] == sub]['readmitted_within_30days'].values[0]), args.seg_len)
        instance_data.append(data_)
    instance_x = []
    instance_y = []
    for ins in instance_data:
        instance_x += ins.data
        instance_y += ins.y
    instance_x, instance_y = shuffle(instance_x, instance_y)
    dataset = EHR_DATASET(instance_x, instance_y, args.enc_in, flag)
    print(flag, len(dataset))
    data_loader = DataLoader(dataset, 
                             batch_size=batch_size, 
                             shuffle=shuffle_flag, 
                             num_workers=args.num_workers, 
                             drop_last=drop_last)
    return instances, data_loader
    # data [((x, mask), instance)] shuffle
    

def image_provider():
    pass

def note_provider():
    pass

def multi_provider():
    pass

