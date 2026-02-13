from data_provider.data_factory import data_provider
from experiments.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
from utils.sam_optimizer import SAM
import os
import time
import warnings
import pdb
import numpy as np
import random

warnings.filterwarnings('ignore')


# train on partial variate data and test on the full variates, used for two types of experiments:
# (1) Generalize on unseen variate (Figure 5 of our paper)
# (2) Efficient training strategy  (Figure 8 of our paper)
class Exp_Long_Term_Forecast_Partial(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Partial, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.optimizer in ['sam', 'asam', 'tse']:
            return SAM(
                self.model.parameters(),
                torch.optim.Adam,
                rho=self.args.sam_rho,
                adaptive=self.args.optimizer in ['asam', 'tse'],
                lr=self.args.learning_rate,
            )
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _load_state_dict_flexible(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            return checkpoint['model']
        return checkpoint

    def _compute_fisher_information(self, model, train_loader, criterion, max_batches):
        fisher = {name: torch.zeros_like(param, device=self.device) for name, param in model.named_parameters()}
        model.train()
        total_batches = 0
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            if i >= max_batches:
                break
            model.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                batch_x_mark = None
                batch_y_mark = None
            else:
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

            partial_start = self.args.partial_start_index
            partial_end = min(self.args.enc_in + partial_start, batch_x.shape[-1])
            batch_x = batch_x[:, :, partial_start:partial_end]
            batch_y = batch_y[:, :, partial_start:partial_end]

            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            if self.args.output_attention:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            elif self.args.channel_independence:
                B, Tx, N = batch_x.shape
                _, Ty, _ = dec_inp.shape
                if batch_x_mark == None:
                    outputs = model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark,
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(B, N, -1).permute(0, 2, 1)
                else:
                    outputs = model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark.repeat(N, 1, 1),
                                    dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark.repeat(N, 1, 1)).reshape(B, N, -1).permute(0, 2, 1)
            else:
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, f_dim:]
            target = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
            loss = criterion(outputs, target)
            loss.backward()
            total_batches += 1
            for name, param in model.named_parameters():
                if param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2)

        if total_batches > 0:
            for name in fisher:
                fisher[name] = fisher[name] / float(total_batches)
        return fisher

    def _prepare_tse_regularization(self, train_loader, criterion):
        if not self.args.tse_parent_a or not self.args.tse_parent_b:
            raise ValueError('TSE requires --tse_parent_a and --tse_parent_b checkpoints.')
        model_a = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        model_b = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        model_a.load_state_dict(self._load_state_dict_flexible(self.args.tse_parent_a), strict=False)
        model_b.load_state_dict(self._load_state_dict_flexible(self.args.tse_parent_b), strict=False)
        params_a = {name: param.detach().clone() for name, param in model_a.named_parameters()}
        params_b = {name: param.detach().clone() for name, param in model_b.named_parameters()}
        fisher_a = self._compute_fisher_information(model_a, train_loader, criterion, self.args.tse_fisher_batches)
        fisher_b = self._compute_fisher_information(model_b, train_loader, criterion, self.args.tse_fisher_batches)
        return params_a, params_b, fisher_a, fisher_b

    def _ewc_loss(self, params_ref, fisher_ref, importance):
        reg = 0.0
        for name, param in self.model.named_parameters():
            if name in fisher_ref:
                reg = reg + torch.sum(fisher_ref[name] * (param - params_ref[name]) ** 2)
        return importance * reg

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, partial_train=False):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                if partial_train:  # we train models with only partial variates from the dataset
                    partial_start = self.args.partial_start_index
                    partial_end = min(self.args.enc_in + partial_start, batch_x.shape[-1])
                    batch_x = batch_x[:, :, partial_start:partial_end]
                    batch_y = batch_y[:, :, partial_start:partial_end]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(
                                B, N, -1).permute(0, 2, 1)
                        else:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                                 batch_x_mark.repeat(N, 1, 1), \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                                 batch_y_mark.repeat(N, 1, 1)) \
                                .reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        tse_state = None
        if self.args.optimizer == 'tse':
            tse_state = self._prepare_tse_regularization(train_loader, criterion)

        train_use_amp = self.args.use_amp and self.args.optimizer == 'adam'
        if train_use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # Variate Generalization training: 
                # We train with partial variates (args.enc_in < number of dataset variates)
                # and test the obtained model directly on all variates.
                partial_start = self.args.partial_start_index
                partial_end = min(self.args.enc_in + partial_start, batch_x.shape[-1])
                batch_x = batch_x[:, :, partial_start:partial_end]
                batch_y = batch_y[:, :, partial_start:partial_end]
                # Efficient training strategy: randomly choose part of the variates
                # and only train the model with selected variates in each batch 
                if self.args.efficient_training:
                    _, _, N = batch_x.shape
                    index = np.stack(random.sample(range(N), N))[-self.args.enc_in:]
                    batch_x = batch_x[:, :, index]
                    batch_y = batch_y[:, :, index]

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if train_use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(
                                B, N, -1).permute(0, 2, 1)
                        else:
                            a = batch_x.permute(0, 2, 1)
                            b = batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1)
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                                 batch_x_mark.repeat(N, 1, 1), \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                                 batch_y_mark.repeat(N, 1, 1)) \
                                .reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.optimizer == 'tse':
                    params_a, params_b, fisher_a, fisher_b = tse_state
                    loss_ewc_a = self._ewc_loss(params_a, fisher_a, self.args.tse_importance)
                    loss_ewc_b = self._ewc_loss(params_b, fisher_b, self.args.tse_importance)
                    total_loss = loss + self.args.tse_alpha * loss_ewc_a + (1.0 - self.args.tse_alpha) * loss_ewc_b
                    total_loss.backward()
                    model_optim.first_step(zero_grad=True)

                    if self.args.output_attention:
                        second_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            second_outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(B, N, -1).permute(0, 2, 1)
                        else:
                            second_outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark.repeat(N, 1, 1), dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark.repeat(N, 1, 1)).reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        second_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    second_outputs = second_outputs[:, -self.args.pred_len:, f_dim:]
                    second_loss = criterion(second_outputs, batch_y)
                    second_loss.backward()
                    model_optim.second_step(zero_grad=True)
                elif self.args.optimizer in ['sam', 'asam']:
                    loss.backward()
                    model_optim.first_step(zero_grad=True)

                    if self.args.output_attention:
                        second_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            second_outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(B, N, -1).permute(0, 2, 1)
                        else:
                            second_outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark.repeat(N, 1, 1), dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark.repeat(N, 1, 1)).reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        second_outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    second_outputs = second_outputs[:, -self.args.pred_len:, f_dim:]
                    second_loss = criterion(second_outputs, batch_y)
                    second_loss.backward()
                    model_optim.second_step(zero_grad=True)
                elif train_use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, partial_train=True)
            test_loss = self.vali(test_data, test_loader, criterion, partial_train=False)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # During model inference, test the obtained model directly on all variates.
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    elif self.args.channel_independence:  # compare the result with channel_independence
                        B, Tx, N = batch_x.shape
                        _, Ty, _ = dec_inp.shape
                        if batch_x_mark == None:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1), batch_x_mark, \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1), batch_y_mark).reshape(
                                B, N, -1).permute(0, 2, 1)
                        else:
                            outputs = self.model(batch_x.permute(0, 2, 1).reshape(B * N, Tx, 1),
                                                 batch_x_mark.repeat(N, 1, 1), \
                                                 dec_inp.permute(0, 2, 1).reshape(B * N, Ty, 1),
                                                 batch_y_mark.repeat(N, 1, 1)) \
                                .reshape(B, N, -1).permute(0, 2, 1)
                    else:
                        # directly test the trained model on all variates without fine-tuning.
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()
                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
