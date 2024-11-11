import numpy as np
import time
import argparse
import torch
import warnings
import os
warnings.filterwarnings("ignore")
from exp.exp_main import Exp_main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification of comatose state")
    
    ## basic config
    parser.add_argument("--mode", type=str, default="train", help="the mode of exp, train or test only")
    parser.add_argument("--root_path", type=str, default="../stat-3612-group-project-2024-fall/", help="the root path of multi-modal data")
    parser.add_argument("--ehr_path", type=str, default="ehr_preprocessed_seq_by_day_cat_embedding.pkl", help="ehr path related to the root path")
    parser.add_argument("--image_path", type=str, default="image", help="image path related to the root path")
    parser.add_argument("--note_path", type=str, default="notes.csv", help="note path related to the root path")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--result_path", type=str, default="./representation")

    ## EHR embedding
    # TODO: parameters used in ehr embedding
    parser.add_argument("--load_ehr", type=bool, default=True)
    parser.add_argument("--seg_len", type=int, default=12, help="the segment length of each instance, padding for less, seperate for more")
    parser.add_argument("--ehr_embed_length", type=int, default=256, help="the embedding length of ehr modal")
    parser.add_argument("--enc_in", type=int, default=171, help="the feature num of ehr series")
    parser.add_argument("--freq", type=str, default='d', help="freqency of ehr data")
    parser.add_argument("--batch_size", type=int, default=125, help="batchsize in ehr model")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--channel_independence", type=bool, default=False)
    parser.add_argument("--output_attention", action='store_true', help='whether to output attention in ecoder', default=True)
    parser.add_argument('--dec_in', type=int, default=171, help="the size of decoder input")
    parser.add_argument("--c_out", type=int, default=2, help="the output size")
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    ## Image embedding
    # TODO: parameters used in image embedding


    ## Clinical Note embedding
    # TODO: parameters used in note embedding


    ## Multi-modal merging
    # TODO: parameters used in multi-modal merging 

    ## Others
    # TODO: some other parameters 
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='2', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
 
    args = parser.parse_args()
    # args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    
    print("Args in experiment:")
    print(args)

    Exp = Exp_main # the main process of experiment
    torch.multiprocessing.set_sharing_strategy("file_system")
    if args.mode == "train":  
        exp = Exp(args)
        time_start = time.time()
        exp.train() # this step only finish the training of predictor
        time_end = time.time()
        print("Training time cost: {}s. ".format(time_end-time_start))
        exp.test()
        print("Inference time cost: {}s".format(time.time() - time_end))
              
    elif args.mode == "test":
        # here is used for test, predictor_loader, classifier_loader = 1, 
        # predictor_load_path, classifier_load_path should be valid, 
        exp = Exp(args)
        
        print(">>>>>>>start testing : >>>>>>>>>>>>>>>>>>>>>>>>>>")
        # time_start = time.time()
        exp.test()
        # time_end = time.time()
        # print("Inference time cost: {}s ".format(time_end-time_start))
        print("Experiment finished!")

