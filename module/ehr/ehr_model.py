# TODO: the model class to train, embed the ehr data
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.ehr.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from module.ehr.layers.SelfAttention_Family import FullAttention, AttentionLayer
from module.ehr.layers.Embed import DataEmbedding
from module.ehr.utils.tools import EarlyStopping, adjust_learning_rate
import numpy as np
import torch.optim as optim
import os
import time
from sklearn.metrics import roc_curve, auc
from utils.tools import dict2pkl


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention

        if configs.channel_independence:
            self.enc_in = 1
            self.dec_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.dec_in = configs.dec_in
            self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(self.enc_in, configs.ehr_embed_length, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.ehr_embed_length, configs.n_heads),
                    configs.ehr_embed_length,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.ehr_embed_length)
        )

        self.projection = nn.Sequential(nn.Linear(configs.ehr_embed_length, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 32), 
                                        nn.ReLU(),
                                        nn.Linear(32, configs.c_out),
                                        nn.Softmax(dim=1))


    def forward(self, x_enc):
        batch_size, sequence_len, input_dim = x_enc.shape
        is_padding = torch.all(x_enc == torch.zeros(input_dim), dim=-1)
        attention_mask = torch.where(is_padding, torch.zeros((batch_size, sequence_len), dtype = torch.bool), 
                             torch.ones((batch_size, sequence_len), dtype = torch.bool))
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=attention_mask)
        enc_out = enc_out[:, -1, :].squeeze(1)
        label = self.projection(enc_out)
        if self.output_attention:
            return enc_out, label, attns
        return enc_out, label



class EHR_Representation():
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.model = self._build_model()

    def _build_model(self):
        model = Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (instance, batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            enc_out, prob, attn = self.model(batch_x)
                        else:
                            enc_out, prob = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        enc_out, prob, attn = self.model(batch_x)
                    else:
                        enc_out, prob = self.model(batch_x)
                loss = criterion(prob, batch_y)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, train_loader, vali_loader):
        path = os.path.join(self.args.checkpoints)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_patient_instance = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            train_steps = 0
            self.model.train()
            epoch_time = time.time()
            iter_count = 0
            train_steps = len(train_loader)
            epoch_time = time.time()
            train_loss = []
            for i, (instance, batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            enc_out, prob, attn = self.model(batch_x)
                        else:
                            enc_out, prob = self.model(batch_x)

                        loss = criterion(prob, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        enc_out, prob, attn = self.model(batch_x)
                    else:
                        enc_out, prob = self.model(batch_x)
                    # print(enc_out.shape, prob.shape, attn[0].shape)
                    loss = criterion(prob, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 5 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
            vali_loss = self.vali(vali_loader, criterion)
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {0}, Patient_instance: {1}, Steps: {2} | Train Loss: {3:.7f} Vali Loss: {4:.7f}".format(
                    epoch + 1, train_patient_instance, train_steps, np.average(train_loss), vali_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # get_cka(self.args, setting, self.model, train_loader, self.device, epoch)

        best_model_path = path + '/' + 'ehr-checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test_loader, instances, save_path=None):
        if self.args.load_ehr:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/', 'ehr-checkpoint.pth')))
        criterion = self._select_criterion()
        probs = np.array([])
        labels = np.array([])
        test_loss = []
        representation = {key: [] for key in instances}
        self.model.eval()
        with torch.no_grad():
            for i, (instance, batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            enc_out, prob, attn = self.model(batch_x)
                        else:
                            enc_out, prob = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        enc_out, prob, attn = self.model(batch_x)

                    else:
                        enc_out, prob = self.model(batch_x)
                enc_out = enc_out.detach().cpu().numpy()
                for j in range(len(instance)):
                    representation[instance[j]].append(enc_out[j])
                loss = criterion(prob, batch_y)
                test_loss.append(loss)
                # print(prob.shape)
                prob = prob[:, 1].detach().cpu().numpy().flatten()
                batch_y = batch_y.detach().cpu().numpy().flatten()
                # print(prob.shape)
                probs = np.concatenate([probs, prob])
                labels = np.concatenate([labels, batch_y])
        result = sorted(list(zip(probs, labels)), key=lambda x: x[1])
        probs = [i[0] for i in result]
        labels = [i[1] for i in result]
        fpr, tpr, thresholds = roc_curve(labels, probs)
        print("EHR Modal Test classification AUROC: {}".format(auc(fpr, tpr)))
        if not save_path == None:
            print(save_path)
            if not os.path.exists(save_path):
                print(save_path)
                os.makedirs(save_path)
            representation = {key: np.mean(value, axis=0) for key, value in representation.items()}
            dict2pkl(representation, save_path + "/representation.pkl")
        return representation


    def represent(self, data_loader, instances, save_path):
        if self.args.load_ehr:
            best_model_path = self.args.checkpoints + '/' + 'ehr-checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))
        representation = {key:[] for key in instances}
        self.model.eval()
        with torch.no_grad():
            for i, (instance, batch_x) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            enc_out, prob, attn = self.model(batch_x)
                        else:
                            enc_out, prob = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        enc_out, prob, attn = self.model(batch_x)
                    else:
                        enc_out, prob = self.model(batch_x)
                enc_out = enc_out.detach().cpu().numpy()
                for j in range(len(instance)):
                    representation[instance[j]].append(enc_out[j])
        representation = {key: np.mean(value, axis=0) for key, value in representation.items()}
        # result save
        folder_path = save_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        dict2pkl(representation, os.path.join(folder_path, "representation.pkl"))
        return representation
