import pickle
import pandas as pd
import numpy as np

def dict2pkl(dictionary:dict, path):
    with open(path, 'wb') as f:
        pickle.dump(dictionary, f)
    print(path)
    return 

def get_instance_readmitted(path, instances):
    """Get the readmitted labels

    Args:
        path (str): the train/val/test file path
    """
    y_label = {}
    readmittd_df = pd.read_csv(path, usecols=['id', 'readmitted_within_30days'])
    raw_y = readmittd_df.drop_duplicates(subset=['id'], keep="first")[['readmitted_within_30days', 'id']]
    for key in instances:
        y_label[key] = int(raw_y[raw_y['id'] == key]['readmitted_within_30days'].values[0])

    return y_label