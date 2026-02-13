from collections import deque
import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate, visual

warnings.filterwarnings('ignore')


class MemoryBank:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.queue = deque(maxlen=capacity)

    def add(self, reps):
        for r in reps.detach():
            self.queue.append(r)

    def tensor(self):
        if not self.queue:
            return None
        return torch.stack(list(self.queue), dim=0).to(self.device)

    def compress(self, k):
        mem = self.tensor()
        if mem is None or mem.shape[0] <= k:
            return torch.tensor(0.0, device=self.device)

        idx = torch.randperm(mem.shape[0], device=self.device)[:k]
        centers = mem[idx].clone()
        for _ in range(5):
            sim = torch.matmul(F.normalize(mem, dim=-1), F.normalize(centers, dim=-1).T)
            assign = sim.argmax(dim=1)
            for i in range(k):
                group = mem[assign == i]
                if group.numel() > 0:
                    centers[i] = group.mean(dim=0)

        loss = ((mem - centers[assign]) ** 2).mean()
        self.queue = deque([c.detach().cpu() for c in centers], maxlen=self.capacity)
        return loss


class Exp_Long_Term_Forecast_DDHACE(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.memory = MemoryBank(args.ddhace_memory_size, self.device)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _time_weight(self, B):
        t = torch.arange(B, device=self.device).float()
        center = B - 1
        return torch.exp(-self.args.ddhace_time_alpha * torch.abs(t - center))

    def _contrastive_loss(self, repr_vec, diff_negs):
        z = F.normalize(repr_vec, dim=-1)
        pos_sim = (z * z).sum(dim=-1) / self.args.ddhace_temperature

        mem = self.memory.tensor()
        neg_sims = []
        if mem is not None:
            mem = F.normalize(mem, dim=-1)
            neg_sims.append(torch.matmul(z, mem.T) / self.args.ddhace_temperature)

        diff_s = torch.einsum('bd,bkd->bk', z, diff_negs) / self.args.ddhace_temperature
        neg_sims.append(diff_s)

        neg_logits = torch.cat(neg_sims, dim=1)
        w = self._time_weight(z.shape[0]).unsqueeze(-1)
        denom = torch.exp(pos_sim) + (w * torch.exp(neg_logits)).sum(dim=1)
        return -torch.log(torch.exp(pos_sim) / (denom + 1e-8)).mean()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y, batch_x_mark, batch_y_mark in vali_loader:
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                total_loss.append(criterion(outputs.detach().cpu(), batch_y.detach().cpu()))

        self.model.train()
        return np.average(total_loss)

    def train(self, setting):
        train_data, train_loader = self._get_data('train')
        vali_data, vali_loader = self._get_data('val')
        test_data, test_loader = self._get_data('test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_loss = []
            self.model.train()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs, aux = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, return_aux=True)
                repr_vec = aux['repr']
                diff_negs = self.model.sample_diffusion_negatives(repr_vec, steps=self.args.ddhace_diff_steps)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                mse_loss = criterion(outputs, batch_y)
                nc_loss = self._contrastive_loss(repr_vec, diff_negs)
                mem_loss = self.memory.compress(self.args.ddhace_mem_clusters) if (i + 1) % self.args.ddhace_mem_update_steps == 0 else torch.tensor(0.0, device=self.device)
                cstr_loss = (repr_vec.norm(dim=-1) - 1.0).abs().mean()
                l2_reg = torch.tensor(0.0, device=self.device)
                for p in self.model.parameters():
                    l2_reg += p.pow(2).sum()

                loss = (
                    self.args.ddhace_lambda_c * nc_loss
                    + self.args.ddhace_lambda_f * mse_loss
                    + self.args.ddhace_lambda_l2 * l2_reg
                    + self.args.ddhace_lambda_mc * mem_loss
                    + self.args.ddhace_lambda_cstr * cstr_loss
                )

                loss.backward()
                model_optim.step()

                self.memory.add(torch.cat([F.normalize(repr_vec, dim=-1), diff_negs.reshape(-1, diff_negs.shape[-1])], dim=0))
                train_loss.append(loss.item())

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]
                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    history = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((history[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((history[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds).reshape(-1, self.args.pred_len, preds[0].shape[-1])
        trues = np.array(trues).reshape(-1, self.args.pred_len, trues[0].shape[-1])
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
