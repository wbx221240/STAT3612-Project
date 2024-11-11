import os
import torch

class Exp_basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.ehr_representation = self._build_ehr_representation()
        self.image_representation = self._build_image_representation()
        self.note_representation = self._build_note_representation()
        self.multi_modal = self._build_multi_modal()
        
        
    def _build_ehr_representation(self):
        raise NotImplementedError
        return None
    
    def _build_image_representation(self):
        raise NotImplementedError
        return None
    
    def _build_note_representation(self):
        raise NotImplementedError
        return None
    
    def _build_multi_modal(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
            
        return device

    def _get_ehr(self):
        pass
    
    def _get_image(self):
        pass
    
    def _get_note(self):
        pass

    def train(self): # used only for train
        pass

    def test(self):
        pass