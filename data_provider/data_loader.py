# TODO: In this file, define the class(es) used in each model. For example, in ehr modal, we need to define
# a Dataset class for the raw ehr data and do some essential transformation. 


from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler

class EHR_DATASET(Dataset):
    def __init__(self, data, y, feature_num, flag):
        self.feature_num = feature_num 
        self.instance = [ins[1] for ins in data]
        self.data = [ins[0] for ins in data]
        self.scaler = self._get_scaler()
        self.y = y
        self.flag = flag

    def __getitem__(self, index):
        if self.flag == "test":
            return self.instance[index], self.data[index]
        return self.instance[index], self.scaler.transform(self.data[index]), self.y[index]
    
    def __len__(self):
        return len(self.data)
    
    def _get_scaler(self):
        scaler = StandardScaler()
        data = np.array(self.data).reshape(-1, self.feature_num)
        scaler.fit(data)
        return scaler

class EHR_Dataset:
    def __init__(self, instance, raw_data, raw_y, seg_len):
        self.raw_x = raw_data
        self.raw_y = raw_y
        self.seg_len = seg_len
        self.x, self.mask, self.y = self._transform_data()
        self.instance = [instance] * self.x.shape[0]
        self.data = list(zip(self.x, self.instance))
        
    
    def _transform_data(self):
        data_complement = []
        mask_complement = []
        seq_len = self.raw_x.shape[0]
        seg_num = seq_len // self.seg_len
        i = 0
        flag = 0
        # print(self.raw_x.shape)
        for i in range(seg_num):
            flag = 1
            data_complement.append(self.raw_x[i * self.seg_len: (i+1) * self.seg_len])
            mask_complement.append(np.ones((self.seg_len, self.raw_x.shape[1])))
        complemented_, mask_complemented = self.padding(self.raw_x[(i+flag)*self.seg_len: ], padding=0)
        data_complement.append(complemented_)
        mask_complement.append(mask_complemented)
        data_complement = np.array(data_complement)
        mask_complement = np.array(mask_complement)
        assert data_complement.shape == mask_complement.shape
        y = np.array([self.raw_y] * data_complement.shape[0]).tolist()
        return data_complement, mask_complement, y

    def padding(self, data, padding):
        # print(data.shape)
        shape = data.shape
        if padding == 0:
            data_padding = np.zeros((self.seg_len - data.shape[0], data.shape[1]))
            data = np.concatenate([data, data_padding], axis=0)
        elif padding == 1:
            data_padding = np.ones((self.seg_len - data.shape[0], data.shape[1]))
            data = np.concatenate([data, data_padding], axis=0)
        mask = np.concatenate([np.ones((shape[0], shape[1])), np.zeros((data.shape[0] - shape[0], shape[1]))], axis=0)
        return data, mask



class Image_Dataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass


class Note_Dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass


class Multi_Dataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self):
        pass


