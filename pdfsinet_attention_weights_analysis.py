import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

from qiantang_model.pd_fsinet import PD_FSINet
from qiantang_model.qiantang_dataloader import Dataset_QiantangTidal

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

class AttentionWeightsAnalyzer:
    
    def __init__(self, device='cuda', pred_len_map=None):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
        if pred_len_map is None:
            self.pred_len_map = {14: 'Medium-term', 60: 'Long-term'}
        else:
            self.pred_len_map = pred_len_map


    def _get_model_paths(self):
        model_files = {
            14: 'pd_v2_fsinet_pl14_best.pth',
            60: 'pd_v2_fsinet_pl60_best.pth'
        }
        return model_files

    def _load_model(self, pred_len):
        model_files = self._get_model_paths()
        model_path = model_files[pred_len]
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not avialable : {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        configs = checkpoint['configs']
        
        model = PD_FSINet(**configs).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Model loaded for pred_len = {pred_len}")
        return model, configs

    def _get_dataloader(self, configs):
        dataset = Dataset_QiantangTidal(
            root_path='./dataset/',
            flag='test',
            size=[configs['seq_len'], 0, configs['pred_len']],
            data_path='cq1_processed.csv',
            scale=False
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        return dataloader

    def extract_attention_weights(self, pred_len):
        model, configs = self._load_model(pred_len)
        dataloader = self._get_dataloader(configs)
        sample_batch = next(iter(dataloader))
        x_hist_dyn, x_hist_per, x_hist_year, y_true_dyn, y_fut_per, y_fut_year = sample_batch
        x_hist_dyn = x_hist_dyn.float().to(self.device)
        x_hist_per = x_hist_per.float().to(self.device)
        x_hist_year = x_hist_year.float().to(self.device)
        y_fut_per = y_fut_per.float().to(self.device)
        y_fut_year = y_fut_year.float().to(self.device)

        all_blocks_attention_weights = []
        
        with torch.no_grad():
            x_hist_per_embedded = model.fourier_embedder(x_hist_per)
            x_per_final_hist = torch.cat([x_hist_per_embedded, x_hist_year], dim=-1)
            astro_full = model.astro_forecaster(x_per_final_hist)
            B_ast_hist = astro_full[:, :model.seq_len, :]
            current_residual = x_hist_dyn - B_ast_hist
            for i in range(len(model.blocks)):
                block = model.blocks[i]
                is_first = (i == 0)

                F_mixed = block.pimdm(current_residual, B_ast_hist if is_first else None, is_first_block=is_first)
                F_fin, p_connects_list, sparse_attention_weights = block.pgtsi_net(
                    F_mixed, return_sparse_attention=True
                )
                attention_weights_numpy = [attn.cpu().detach().numpy() for attn in sparse_attention_weights]
                for j, attn in enumerate(attention_weights_numpy):
                    print(f"      Group {j}: shape = {attn.shape}, sparsity = {(attn == 0).sum() / attn.size * 100:.1f}%")
                
                all_blocks_attention_weights.append(attention_weights_numpy)
                Y_comp = block.prediction_head(F_fin)
                X_rec = block.backcast_head(F_fin)
                current_residual = current_residual - X_rec

        return all_blocks_attention_weights

    def plot_attention_weights(self, all_data):
        num_blocks = len(next(iter(all_data.values())))
        pred_lens = list(self.pred_len_map.keys())
        num_groups = 2  
        selected_groups = [0, 2]  # Potential-1 和 Kinetic-1
        
        fig, axes = plt.subplots(num_blocks, len(pred_lens) * num_groups, 
                                 figsize=(12, 8),
                                 gridspec_kw={'hspace': 0.01, 'wspace': 0.25})
        
        group_names = ['Potential Group', 'Dynamic Group']

        seq_len = 96
        freq_len = seq_len // 2 + 1  # 49
        freq_labels = []
        freq_ticks = []
        for i in range(0, freq_len, 10):
            freq_labels.append(f'f{i}')
            freq_ticks.append(i)
        if freq_len - 1 not in freq_ticks:
            freq_labels.append(f'f{freq_len-1}')
            freq_ticks.append(freq_len - 1)

        for i in range(num_blocks):  # 行：Blocks
            for j, pred_len in enumerate(pred_lens): 
                for k in range(num_groups):  
                    ax_idx = j * num_groups + k
                    ax = axes[i, ax_idx]
                    actual_group_idx = selected_groups[k]
                    
                    if (pred_len not in all_data or i >= len(all_data[pred_len]) or 
                        actual_group_idx >= len(all_data[pred_len][i])):
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    attention_matrix = all_data[pred_len][i][actual_group_idx]
                    sparsity = (attention_matrix == 0).sum() / attention_matrix.size * 100
                    im = ax.imshow(attention_matrix, cmap='viridis', vmin=0, vmax=attention_matrix.max(), 
                                   aspect='equal', origin='upper')

                    cbar = fig.colorbar(im, ax=ax, pad=0.05, shrink=0.8, aspect=20)
                    if ax_idx == len(pred_lens) * num_groups - 1:
                        cbar.set_label('Attention Weight', fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                    
                    if i == 0:
                        ax.set_title(f'PL={pred_len}\n{group_names[k]}', fontsize=14)

                    if j == 0 and k == 0:
                        ax.set_ylabel(f'Block {i+1}', fontsize=14, labelpad=5)

                    ax.set_xticks(freq_ticks)
                    ax.set_yticks(freq_ticks)

                    ax.set_xticklabels(freq_labels, rotation=45, ha='right', fontsize=12)
                    ax.set_yticklabels(freq_labels, rotation=0, fontsize=12)

        plt.tight_layout(h_pad=0.05, w_pad=0.5)
        save_path = 'sparse_attention_weights_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.show()

    def run(self):
    
        all_results = {}
        for pred_len in self.pred_len_map.keys():
            try:
                all_results[pred_len] = self.extract_attention_weights(pred_len)

            except Exception as e:

                continue
        
        if all_results:
            self.plot_attention_weights(all_results)

if __name__ == '__main__':
    analyzer = AttentionWeightsAnalyzer(pred_len_map={14: 'Medium-term', 60: 'Long-term'})
    analyzer.run()
