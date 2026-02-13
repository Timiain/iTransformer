import argparse
import os
import random
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from model import iTransformer
from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


plt.switch_backend('agg')


@dataclass
class DatasetConfig:
    name: str
    data: str
    root_path: str
    data_path: str
    freq: str


DEFAULT_DATASETS = [
    DatasetConfig('ETTh1', 'ETTh1', './dataset/ETT-small/', 'ETTh1.csv', 'h'),
    DatasetConfig('ETTh2', 'ETTh2', './dataset/ETT-small/', 'ETTh2.csv', 'h'),
    DatasetConfig('ETTm1', 'ETTm1', './dataset/ETT-small/', 'ETTm1.csv', 't'),
    DatasetConfig('ETTm2', 'ETTm2', './dataset/ETT-small/', 'ETTm2.csv', 't'),
    DatasetConfig('Electricity', 'custom', './dataset/electricity/', 'electricity.csv', 'h'),
    DatasetConfig('Traffic', 'custom', './dataset/traffic/', 'traffic.csv', 'h'),
    DatasetConfig('Weather', 'custom', './dataset/weather/', 'weather.csv', 'h'),
    DatasetConfig('Exchange', 'custom', './dataset/exchange_rate/', 'exchange_rate.csv', 'd'),
    DatasetConfig('Solar', 'Solar', './dataset/Solar/', 'solar_AL.txt', 'h'),
]


class MultiDatasetExperiment:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = iTransformer.Model(args).float().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu)
            print(f'Use GPU: cuda:{self.args.gpu}')
            return torch.device(f'cuda:{self.args.gpu}')
        print('Use CPU')
        return torch.device('cpu')

    @staticmethod
    def _set_dataset_args(args, cfg: DatasetConfig):
        args.data = cfg.data
        args.root_path = cfg.root_path
        args.data_path = cfg.data_path
        args.freq = cfg.freq

    @staticmethod
    def _need_time_mark(dataset_name: str, data_flag: str):
        return ('PEMS' not in dataset_name) and ('Solar' not in data_flag)

    def _forward_and_loss(self, batch, dataset_name, data_flag):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        if self._need_time_mark(dataset_name, data_flag):
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
        else:
            batch_x_mark = None
            batch_y_mark = None

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        f_dim = -1 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        target = batch_y[:, -self.args.pred_len:, f_dim:]

        loss = self.criterion(outputs, target)
        return loss, outputs.detach().cpu().numpy(), target.detach().cpu().numpy()

    def build_loaders(self, dataset_cfgs: List[DatasetConfig]):
        bundles = []
        for cfg in dataset_cfgs:
            self._set_dataset_args(self.args, cfg)
            train_data, train_loader = data_provider(self.args, flag='train')
            val_data, val_loader = data_provider(self.args, flag='val')
            test_data, test_loader = data_provider(self.args, flag='test')
            bundles.append({
                'cfg': cfg,
                'train_data': train_data,
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
            })
        return bundles

    def validate(self, bundles):
        self.model.eval()
        val_losses = {}
        with torch.no_grad():
            for bundle in bundles:
                cfg = bundle['cfg']
                losses = []
                for batch in bundle['val_loader']:
                    loss, _, _ = self._forward_and_loss(batch, cfg.name, cfg.data)
                    losses.append(loss.item())
                val_losses[cfg.name] = float(np.mean(losses)) if losses else np.inf
        self.model.train()
        return val_losses

    def train(self, bundles, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(1, self.args.train_epochs + 1):
            epoch_losses = []
            random.shuffle(bundles)
            self.model.train()

            for bundle in bundles:
                cfg = bundle['cfg']
                for batch in bundle['train_loader']:
                    self.optimizer.zero_grad()
                    loss, _, _ = self._forward_and_loss(batch, cfg.name, cfg.data)
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())

            val_losses = self.validate(bundles)
            avg_train_loss = float(np.mean(epoch_losses)) if epoch_losses else np.inf
            avg_val_loss = float(np.mean(list(val_losses.values())))

            details = ', '.join([f"{k}: {v:.6f}" for k, v in val_losses.items()])
            print(f'Epoch {epoch}/{self.args.train_epochs} | train_loss={avg_train_loss:.6f} | avg_val_loss={avg_val_loss:.6f}')
            print(f'Validation by dataset -> {details}')

            early_stopping(avg_val_loss, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                print('Early stopping triggered.')
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.args)

        best_model_path = os.path.join(checkpoint_dir, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

    def test_all(self, bundles):
        rows = []
        self.model.eval()
        with torch.no_grad():
            for bundle in bundles:
                cfg = bundle['cfg']
                preds, trues = [], []
                for batch in bundle['test_loader']:
                    _, pred, true = self._forward_and_loss(batch, cfg.name, cfg.data)
                    preds.append(pred)
                    trues.append(true)

                preds = np.array(preds).reshape(-1, self.args.pred_len, preds[0].shape[-1])
                trues = np.array(trues).reshape(-1, self.args.pred_len, trues[0].shape[-1])
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                rows.append({
                    'dataset': cfg.name,
                    'mae': mae,
                    'mse': mse,
                    'rmse': rmse,
                    'mape': mape,
                    'mspe': mspe,
                })
                print(f'[Test] {cfg.name}: mse={mse:.6f}, mae={mae:.6f}')

        df = pd.DataFrame(rows)
        avg_row = {'dataset': 'Average'}
        for c in ['mae', 'mse', 'rmse', 'mape', 'mspe']:
            avg_row[c] = float(df[c].mean())
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)
        return df


def plot_radar(df: pd.DataFrame, output_path: str):
    data = df[df['dataset'] != 'Average'].copy()
    labels = data['dataset'].tolist()
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(polar=True))

    for ax, metric_name in zip(axes, ['mse', 'mae']):
        values = data[metric_name].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=metric_name.upper())
        ax.fill(angles, values, alpha=0.2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(f'Joint iTransformer ({metric_name.upper()})')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Joint training on all datasets with iTransformer')
    parser.add_argument('--model_id', type=str, default='joint_all_datasets_96_96')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results/joint_all_datasets')

    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--class_strategy', type=str, default='projection')
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--use_norm', type=int, default=1)

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lradj', type=str, default='type1')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)

    return parser.parse_args()


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.checkpoints, args.model_id)

    exp = MultiDatasetExperiment(args)
    bundles = exp.build_loaders(DEFAULT_DATASETS)
    exp.train(bundles, checkpoint_dir)
    df = exp.test_all(bundles)

    csv_path = os.path.join(args.results_dir, 'joint_metrics.csv')
    radar_path = os.path.join(args.results_dir, 'joint_radar.png')
    df.to_csv(csv_path, index=False)
    plot_radar(df, radar_path)

    print(f'Saved metrics csv to: {csv_path}')
    print(f'Saved radar figure to: {radar_path}')


if __name__ == '__main__':
    main()
