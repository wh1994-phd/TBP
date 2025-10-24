import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import os

from qiantang_model.pd_fsinet import PD_FSINet
from qiantang_model.qiantang_dataloader import Dataset_QiantangTidal
from qiantang_model.loss import CompositeLoss
from utils.timefeatures import FourierFeatureEmbedding
from qiantang_model.astro_forecaster import AstroForecaster


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PD_FSINet_Lightning(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        fourier_feature_info = {
            'Day_of_Year': 366,
            'Lunar_Day': 30, 
            'Month': 12
        }
        self.fourier_embedder = FourierFeatureEmbedding(
            feature_info=fourier_feature_info,
            embed_dim=self.hparams.embed_dim
        )
        
        self.model = PD_FSINet(
            num_blocks=self.hparams.num_blocks,
            seq_len=self.hparams.seq_len,
            pred_len=self.hparams.pred_len,
            in_channels_dyn=self.hparams.in_channels_dyn,
            in_channels_per=self.hparams.in_channels_per,
            d_model=self.hparams.d_model,
            n_heads=self.hparams.n_heads,
            d_ff=self.hparams.d_ff,
            dropout=self.hparams.dropout,
            num_bands=self.hparams.num_bands,
            d_mem=self.hparams.d_mem,
            embed_dim=self.hparams.embed_dim
        )
        
        self.criterion = CompositeLoss(
            lambda_sdd=self.hparams.lambda_sdd,
            lambda_sparse=self.hparams.lambda_sparse,
            lambda_smooth=self.hparams.lambda_smooth,
            moving_avg_kernel=getattr(self.hparams, 'moving_avg_kernel', 7)
        )
        
        self.validation_step_outputs = []


    def forward(self, x_hist_dyn, x_hist_per, x_hist_year, y_fut_per, y_fut_year):
        return self.model(x_hist_dyn, x_hist_per, x_hist_year, y_fut_per, y_fut_year)

    def training_step(self, batch, batch_idx):
        x_hist_dyn, x_hist_per, x_hist_year, y_true_dyn, y_fut_per, y_fut_year = batch
        outputs = self.model(x_hist_dyn, x_hist_per, x_hist_year, y_fut_per, y_fut_year)
        total_loss, loss_dict = self.criterion(outputs, y_true_dyn, x_hist_dyn)
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for loss_name, loss_value in loss_dict.items():
            if loss_name != 'total_loss':
                self.log(f'train_{loss_name}', loss_value, on_step=True, on_epoch=True) 
        self.log('train_L_smooth', loss_dict['L_smooth'], on_step=True, on_epoch=True)
        train_mse = torch.mean((outputs['prediction'] - y_true_dyn) ** 2)
        self.log('train_mse', train_mse, on_step=False, on_epoch=True, prog_bar=True)
        if not hasattr(self, 'epoch_stats'):
            self.epoch_stats = {'smooth_losses': [], 'astro_stds': [], 'astro_ranges': [], 'train_mses': []}
        astro_baseline_hist = outputs.get('astro_baseline_hist', None)
        if astro_baseline_hist is not None:
            astro_std = astro_baseline_hist.std().item()
            astro_range = (astro_baseline_hist.max() - astro_baseline_hist.min()).item()
            self.epoch_stats['smooth_losses'].append(loss_dict['L_smooth'].item())
            self.epoch_stats['astro_stds'].append(astro_std)
            self.epoch_stats['astro_ranges'].append(astro_range)
            self.epoch_stats['train_mses'].append(train_mse.item())
      
        return total_loss

    def on_train_epoch_end(self):
        if hasattr(self, 'epoch_stats') and self.epoch_stats['smooth_losses']:
            import numpy as np
            avg_smooth = np.mean(self.epoch_stats['smooth_losses'])
            avg_astro_std = np.mean(self.epoch_stats['astro_stds'])
            avg_astro_range = np.mean(self.epoch_stats['astro_ranges'])
            avg_train_mse = np.mean(self.epoch_stats['train_mses'])
            print(f"Epoch {self.current_epoch}: Train_MSE={avg_train_mse:.6f}, "
                  f"L_smooth={avg_smooth:.6f}, astro_std={avg_astro_std:.6f}, astro_range={avg_astro_range:.6f}")
            self.epoch_stats = {'smooth_losses': [], 'astro_stds': [], 'astro_ranges': [], 'train_mses': []}

    def validation_step(self, batch, batch_idx):
        x_hist_dyn, x_hist_per, x_hist_year, y_true_dyn, y_fut_per, y_fut_year = batch
        
        outputs = self.model(x_hist_dyn, x_hist_per, x_hist_year, y_fut_per, y_fut_year)
        y_pred = outputs['prediction']
        total_loss, loss_dict = self.criterion(outputs, y_true_dyn, x_hist_dyn)
        mse = torch.mean((y_pred - y_true_dyn) ** 2)
        mae = torch.mean(torch.abs(y_pred - y_true_dyn))
        y_mean = torch.mean(y_true_dyn)
        ss_res = torch.sum((y_true_dyn - y_pred) ** 2)
        ss_tot = torch.sum((y_true_dyn - y_mean) ** 2)
        nse = 1 - ss_res / (ss_tot + 1e-8)
        self.validation_step_outputs.append({
            'val_loss': total_loss,
            'val_mse': mse,
            'val_mae': mae,
            'val_nse': nse
        })
        return total_loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
            avg_mse = torch.stack([x['val_mse'] for x in self.validation_step_outputs]).mean()
            avg_mae = torch.stack([x['val_mae'] for x in self.validation_step_outputs]).mean()
            avg_nse = torch.stack([x['val_nse'] for x in self.validation_step_outputs]).mean()
            
            self.log('val_loss', avg_loss, prog_bar=True)
            self.log('val_mse', avg_mse, prog_bar=True)
            self.log('val_mae', avg_mae, prog_bar=True)
            self.log('val_nse', avg_nse, prog_bar=True)
            self.validation_step_outputs.clear()


def create_dataloaders(args):
    train_dataset = Dataset_QiantangTidal(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, 0, args.pred_len],
        scale=True
    )
    
    val_dataset = Dataset_QiantangTidal(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='val',
        size=[args.seq_len, 0, args.pred_len],
        scale=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='PD-FSINet Training')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--max_epochs', type=int, default=50, help='max training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--gradient_clip_val', type=float, default=None, help='gradient clipping value')
    parser.add_argument('--num_blocks', type=int, default=3, help='number of PD-FSINet blocks')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=30, help='prediction sequence length')
    parser.add_argument('--in_channels_dyn', type=int, default=4, help='number of dynamic input channels')
    parser.add_argument('--in_channels_per', type=int, default=3, help='number of periodic input channels')
    parser.add_argument('--d_model', type=int, default=128, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256, help='feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_bands', type=int, default=4, help='number of frequency bands')
    parser.add_argument('--d_mem', type=int, default=32, help='memory dimension')
    parser.add_argument('--embed_dim', type=int, default=16, help='fourier embedding dimension')
    parser.add_argument('--lambda_sdd', type=float, default=1.0, help='SDD loss weight')
    parser.add_argument('--lambda_sparse', type=float, default=0.01, help='sparse loss weight')
    parser.add_argument('--lambda_smooth', type=float, default=0.5, help='smooth loss weight')
    parser.add_argument('--moving_avg_kernel', type=int, default=7, help='moving average kernel size for smooth loss')
    parser.add_argument('--accelerator', type=str, default='auto', help='accelerator')
    parser.add_argument('--devices', type=str, default='auto', help='devices')
    parser.add_argument('--strategy', type=str, default='auto', help='strategy')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='path to checkpoint file to resume training')
    parser.add_argument('--auto_resume', type=bool, default=False, help='auto resume from latest checkpoint')
    args = parser.parse_args()
    
    set_seed(args.seed)
    pl.seed_everything(args.seed, workers=True)  
    for key, value in vars(args).items():
        print(f" {key}: {value}")
    print("=" * 80)
    
    train_loader, val_loader = create_dataloaders(args)
    
    model = PD_FSINet_Lightning(**vars(args))
    
    checkpoint_dir = f"checkpoints/PD-FSINet_NEW_model_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_bs{args.batch_size}_lr{args.learning_rate}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epoch={epoch}-val_mse={val_mse:.6f}',
        monitor='val_mse',
        mode='min',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    
    early_stopping = EarlyStopping(
        monitor='val_mse',
        patience=5,
        mode='min',
        verbose=True
    )

  
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"PD-FSINet_sl{args.seq_len}_pl{args.pred_len}",
        version=None
    )
    
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        callbacks=[checkpoint_callback, early_stopping],
        logger=logger,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val is not None else 0.0,
        deterministic=True,
        enable_progress_bar=True,
        log_every_n_steps=50
    )
    
    if args.resume_from_checkpoint:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_from_checkpoint)
    elif args.auto_resume:
        trainer.fit(model, train_loader, val_loader, ckpt_path='last')
    else:
        trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    main()
