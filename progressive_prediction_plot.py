
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from qiantang_model.pd_fsinet import PD_FSINet
from qiantang_model.qiantang_dataloader import Dataset_QiantangTidal

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

class PDFSINetProgressivePredictionAnalyzer:

    def __init__(self, pred_len=60, sample_idx=250, device='cuda'):
        self.pred_len = pred_len
        self.sample_idx = sample_idx
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.model_path = f'pd_v2_fsinet_pl{pred_len}_best.pth'
        
        self.feature_names = ['LL', 'HL', 'RT', 'CT']
        self.feature_full_names = [
            'Low Tide Water Level', 
            'High Tide Water Level',
            'Rising Time Duration',
            'Tidal Cycle Duration'
        ]
        self.feature_units = ['(m)', '(m)', '(h)', '(h)']
        
        self.progressive_predictions = {}
        
        
        self._load_model_and_data()
    
    def _load_model_and_data(self):

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model mo exist: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        configs = checkpoint['configs']
        self.seq_len = configs['seq_len']

        self.model = PD_FSINet(**configs).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        dataset = Dataset_QiantangTidal(
            root_path='./dataset/',
            flag='test',
            size=[configs['seq_len'], 0, configs['pred_len']],
            data_path='cq1_processed.csv',
            scale=False
        )
        
        sample_data = dataset[self.sample_idx]
        self.x_hist_dyn, self.x_hist_per, self.x_hist_year, \
        self.y_true_dyn, self.y_fut_per, self.y_fut_year = [
            torch.tensor(x).unsqueeze(0).float().to(self.device) for x in sample_data
        ]

    
    def extract_progressive_predictions(self):
        block_states = {}
        
        def create_block_hook(block_idx):
            def hook(module, input, output):
                Y_comp, X_res, p_connects = output
                block_states[f'block_{block_idx}_Y_comp'] = Y_comp.detach().cpu()
                block_states[f'block_{block_idx}_X_res'] = X_res.detach().cpu()
                print(f" Block{block_idx} 捕获: Y_comp={Y_comp.shape}, X_res={X_res.shape}")
            return hook

        hooks = []
        for i in range(len(self.model.blocks)):
            hook = self.model.blocks[i].register_forward_hook(create_block_hook(i+1))
            hooks.append(hook)
        
        try:
            with torch.no_grad():
                outputs = self.model(
                    self.x_hist_dyn, self.x_hist_per, self.x_hist_year,
                    self.y_fut_per, self.y_fut_year
                )
                
                final_prediction = outputs['prediction'].detach().cpu()

                if "astro_baseline_full" in outputs:
                    astro_baseline_full = outputs["astro_baseline_full"].detach().cpu()
                    astro_fut_prediction = astro_baseline_full[:, self.seq_len:, :]
                
                block_predictions = []
                cumulative_prediction = astro_fut_prediction.clone()
              
                for i in range(len(self.model.blocks)):
                    block_key = f'block_{i+1}_Y_comp'
                    if block_key in block_states:
                        Y_comp = block_states[block_key]
                        cumulative_prediction = cumulative_prediction + Y_comp
                        block_predictions.append(cumulative_prediction.clone())
                      
                self.progressive_predictions = {
                    'astro_baseline': astro_fut_prediction,
                    'block1_cumulative': block_predictions[0],
                    'block2_cumulative': block_predictions[1],
                    'block3_cumulative': block_predictions[2],
                    'final_prediction': final_prediction,
                    'y_true': self.y_true_dyn.detach().cpu()
                }
                
        finally:
            for hook in hooks:
                hook.remove()
    
    def _print_prediction_summary(self):

        for key, value in self.progressive_predictions.items():
            if value is not None:
                print(f"{key}: {value.shape}")
    
    def visualize_progressive_predictions(self, save_dir='./progressive_prediction_analysis'):
        
        os.makedirs(save_dir, exist_ok=True)

        astro_baseline = self.progressive_predictions['astro_baseline'][0].numpy()  # [H, C]
        block1_pred = self.progressive_predictions['block1_cumulative'][0].numpy()  # [H, C]
        block2_pred = self.progressive_predictions['block2_cumulative'][0].numpy()  # [H, C]
        block3_pred = self.progressive_predictions['block3_cumulative'][0].numpy()  # [H, C]

        time_steps = np.arange(len(astro_baseline))
        

        fig, axes = plt.subplots(4, 3, figsize=(12, 5))

        predictions = [block1_pred, block2_pred, block3_pred]
        column_titles = ['Block 1 Prediction', 'Block 1+2 Prediction', 'Block 1+2+3 Prediction']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

        for feature_idx in range(4):
            for block_idx in range(3):
                ax = axes[feature_idx, block_idx]

                ax.plot(time_steps, astro_baseline[:, feature_idx], 
                       color='lightblue', linewidth=2, alpha=0.6, 
                       label='Astronomical Baseline', linestyle='--')
                ax.plot(time_steps, predictions[block_idx][:, feature_idx], 
                       color=colors[block_idx], linewidth=3, alpha=0.8,
                       label=f'Cumulative Prediction')
                if block_idx > 0:
                    prev_pred = predictions[block_idx-1][:, feature_idx]
                    curr_pred = predictions[block_idx][:, feature_idx]
                    increment = curr_pred - prev_pred
                    ax.fill_between(time_steps, prev_pred, curr_pred, 
                                   color=colors[block_idx], alpha=0.3,
                                   label=f'Block {block_idx+1} Increment')
                if feature_idx == 0:
                    ax.set_title(column_titles[block_idx], fontsize=14, fontweight='bold')
                if block_idx == 0:
                    ax.set_ylabel(f'{self.feature_names[feature_idx]} {self.feature_units[feature_idx]}', 
                                 fontsize=12, fontweight='bold')
                if feature_idx == 3:
                    ax.set_xlabel('Future Tidal Event Index', fontsize=12)
                

                ax.grid(True, alpha=0.3, linestyle='--')
                

                if feature_idx == 0:
                    ax.legend(fontsize=8, loc='upper right')
                

                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                ax.tick_params(axis='both', which='major', labelsize=10)
                

                if feature_idx < 2:  
                    y_min = min(astro_baseline[:, feature_idx].min(), 
                               predictions[2][:, feature_idx].min()) - 0.1
                    y_max = max(astro_baseline[:, feature_idx].max(), 
                               predictions[2][:, feature_idx].max()) + 0.1
                else:  
                    y_min = min(astro_baseline[:, feature_idx].min(), 
                               predictions[2][:, feature_idx].min()) - 0.5
                    y_max = max(astro_baseline[:, feature_idx].max(), 
                               predictions[2][:, feature_idx].max()) + 0.5
                
                ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'progressive_predictions_4x3_pl{self.pred_len}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_individual_block_contributions(self, save_dir='./progressive_prediction_analysis'):
        
        os.makedirs(save_dir, exist_ok=True)
        astro_baseline = self.progressive_predictions['astro_baseline'][0].numpy()  # [H, C]
        block1_cumulative = self.progressive_predictions['block1_cumulative'][0].numpy()  # [H, C]
        block2_cumulative = self.progressive_predictions['block2_cumulative'][0].numpy()  # [H, C]
        block3_cumulative = self.progressive_predictions['block3_cumulative'][0].numpy()  # [H, C]

        block1_contribution = block1_cumulative - astro_baseline  
        block2_contribution = block2_cumulative - block1_cumulative  
        block3_contribution = block3_cumulative - block2_cumulative  
        time_steps = np.arange(len(astro_baseline))
        

        fig, axes = plt.subplots(4, 3, figsize=(12, 5))

        contributions = [block1_contribution, block2_contribution, block3_contribution]
        column_titles = ['Block 1 Contribution', 'Block 2 Contribution', 'Block 3 Contribution']
        colors = ['#e74c3c', '#f39c12', '#27ae60'] 
        
        for feature_idx in range(4):
            for block_idx in range(3):
                ax = axes[feature_idx, block_idx]
                ax.axhline(y=0, color='black', linewidth=1, alpha=0.3, linestyle='--', label='Zero Baseline')

                contribution_data = contributions[block_idx][:, feature_idx]
                ax.plot(time_steps, contribution_data, 
                       color=colors[block_idx], linewidth=3, alpha=0.8,
                       label=f'Block {block_idx+1} Prediction')
                positive_mask = contribution_data >= 0
                negative_mask = contribution_data < 0
                if np.any(positive_mask):
                    ax.fill_between(time_steps, 0, contribution_data, 
                                   where=positive_mask, 
                                   color=colors[block_idx], alpha=0.3,
                                   label='Positive Contribution')
                if np.any(negative_mask):
                    ax.fill_between(time_steps, 0, contribution_data,
                                   where=negative_mask,
                                   color=colors[block_idx], alpha=0.2)

                if feature_idx == 0:
                    ax.set_title(column_titles[block_idx], fontsize=14, fontweight='bold')

                if block_idx == 0:
                    ax.set_ylabel(f'{self.feature_names[feature_idx]} Δ{self.feature_units[feature_idx]}', 
                                 fontsize=12, fontweight='bold')
                if feature_idx == 3:
                    ax.set_xlabel('Future Tidal Event Index', fontsize=12)

                ax.grid(True, alpha=0.3, linestyle='--')
                

                if feature_idx == 0:
                    ax.legend(fontsize=8, loc='upper right')
                for spine in ax.spines.values():
                    spine.set_linewidth(1.2)
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                y_max = max(abs(contribution_data.min()), abs(contribution_data.max()))
                if y_max > 0:
                    y_margin = y_max * 0.1
                    ax.set_ylim(-y_max - y_margin, y_max + y_margin)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'individual_block_contributions_4x3_pl{self.pred_len}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')

        plt.show()
    
    def run_analysis(self):
        
        try:
            self.extract_progressive_predictions()
            
            self.visualize_progressive_predictions()
            
            self.visualize_individual_block_contributions()

            return self.progressive_predictions
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

def main():
    analyzer = PDFSINetProgressivePredictionAnalyzer(pred_len=60, sample_idx=250)
    progressive_predictions = analyzer.run_analysis()

if __name__ == "__main__":
    main()
