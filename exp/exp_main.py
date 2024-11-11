from data_provider.data_factory import ehr_provider
from exp.exp_basic import Exp_basic
from module.ehr.ehr_model import EHR_Representation
from module.image import image_model
from module.note import note_model
from module.multi import multi_model
from sklearn.preprocessing import StandardScaler
from evaluate.evaluation import Evaluation
import random, os
import pandas as pd
import torch
from utils.tools import get_instance_readmitted

class Exp_main(Exp_basic):
    def __init__(self, args):
        super(Exp_main, self).__init__(args)
        

    def _build_ehr_representation(self):
        # TODO: the model for ehr representation, use this model
        # to get the ehr representation of one patient
        return EHR_Representation(self.args, self.device)
    
    def _build_image_representation(self):
        # TODO: the model for image representation, use this model to 
        # get the image representation of one patient
        return None
    
    def _build_note_representation(self):
        # TODO: the model for note representation, use this
        # model to get the clinical note representation of 
        # one patient
        return None
    
    def _build_multi_modal(self):
        # TODO: the model for multi-modal concatenation and training
        return None
    
    # The next three methods for retrieving data from folders
    # flag used for ["train", "val", "test"]
    def _get_ehr(self, flag):
        # TODO: get ehr data from folder
        instances, dataloader = ehr_provider(self.args, flag)
        return instances, dataloader
    
    def _get_image(self, flag):
        # TODO: get image data from folder
        return None
    
    def _get_note(self, flag):
        # TODO: get note data from folder
        return None

    def train(self):
        """Training method, sequentially process the repesentation of each modal, and
        eventually combine them into a new one for each instance(patient)

        In the end, we will train the multi-modal model on the concatenated representation.

        All the single modal should return the representation of one instance indexed by 
        the patient id? Single modal evaluation can be done if possible.
        """
        ## module1: ehr modal
        # TODO: load ehr from folder and train the ehr model to get the representation
        train_instances, train_loader = self._get_ehr(flag="train")
        vali_instances, vali_loader = self._get_ehr(flag="vali")
        print(">>>>>>>start training ehr representation model: >>>>>>>>>>>>>>>>>>>>>>>>>>")
        # self.ehr_representation.train(train_loader, vali_loader)
        train_ehr_representation = self.ehr_representation.test(train_loader, train_instances, self.args.result_path+'/train')
        val_ehr_representation = self.ehr_representation.test(vali_loader, vali_instances, self.args.result_path+'/valid')
        ehr_evaluation_train = Evaluation(base="LR", setting="EHR MODAL TRAIN", instances=train_instances, path=os.path.join(self.args.root_path, "train.csv"), representation=train_ehr_representation)
        ehr_evaluation_train.evaluate(val_ehr_representation, os.path.join(self.args.root_path, 'valid.csv'), vali_instances)
        ## module2: image modal
        # TODO: load image from folder and train the image model to get the representation

        ## module3: note modal
        # TODO: load note from folder and train the note model to get the representation

        ## module4: multi-modal
        # TODO: concatenate the representation from the above model and train the multi-modal
        # model to finally get the readimission prediction. No test procedure included in this part
        return self


    def test(self):
        y_pred = None
        ## module1: ehr modal
        # TODO: load test ehr data from folder and use the ehr model pre-trained to get the 
        # representation

        ## module2: image modal
        # TODO: load test image data from folder and use the image model pre-trained to get the representation
        # representation

        ## module3: note modal
        # TODO: load test note data from folder and use the note model pre-trained to get
        # representation

        ## module4: multi-modal
        # TODO: concatenate all the three modal representation and use the multi-modal model pre-trained
        # to get the readmission prediction

        #### should return the prediction of readmission 
        ### and save it to a csv
        return y_pred