
import torch
import torch.nn as nn
import torch.nn.functional as F


class JSDivergence(nn.Module):

    def __init__(self, reduction='mean'):
        super(JSDivergence, self).__init__()
        self.kl_div = nn.KLDivLoss(reduction='batchmean', log_target=False)
        self.reduction = reduction

    def forward(self, p, q):
        m = 0.5 * (p + q)
        jsd = 0.5 * (self.kl_div(torch.log(p + 1e-8), m) + self.kl_div(torch.log(q + 1e-8), m))

        if self.reduction == 'mean':
            return jsd.mean()
        elif self.reduction == 'sum':
            return jsd.sum()
        else:
            return jsd


class CompositeLoss(nn.Module):
    def __init__(self, lambda_sdd=1.0, lambda_sparse=0.01, lambda_smooth=0.5, lambda_smooth_pred=0.1,
                 eps_rel=1e-2, eps_sdd=1e-2, moving_avg_kernel=7):
        super(CompositeLoss, self).__init__()
        self.lambda_sdd = lambda_sdd
        self.lambda_sparse = lambda_sparse
        self.lambda_smooth = lambda_smooth  
        self.lambda_smooth_pred = lambda_smooth_pred  
        self.eps_rel = eps_rel
        self.eps_sdd = eps_sdd
        self.jsd_loss = JSDivergence()
        self.moving_avg = nn.AvgPool1d(kernel_size=moving_avg_kernel, stride=1, padding=(moving_avg_kernel - 1) // 2)

    def forward(self, model_outputs, y_true, x_hist):
        y_pred = model_outputs["prediction"]
        b_ast_hist = model_outputs["astro_baseline_hist"]
        sparse_probs = model_outputs["sparse_probs"]

        B, H, C = y_pred.shape
        L = x_hist.shape[1]

        y_pred_freq = torch.fft.rfft(y_pred.permute(0, 2, 1), dim=-1)  # [B, C, H//2+1]
        y_true_freq = torch.fft.rfft(y_true.permute(0, 2, 1), dim=-1)  # [B, C, H//2+1]

        freq_err = torch.abs(y_pred_freq - y_true_freq)
        freq_mag_true = torch.abs(y_true_freq)
        l_fwce_rel = torch.mean(freq_err)

        psd_pred = torch.abs(y_pred_freq) ** 2
        psd_true = torch.abs(y_true_freq) ** 2

        psd_pred_norm = (psd_pred + self.eps_sdd) / torch.sum(psd_pred + self.eps_sdd, dim=-1, keepdim=True)
        psd_true_norm = (psd_true + self.eps_sdd) / torch.sum(psd_true + self.eps_sdd, dim=-1, keepdim=True)

        l_sdd = self.jsd_loss(psd_pred_norm, psd_true_norm)

        l_pred = l_fwce_rel + self.lambda_sdd * l_sdd

        if sparse_probs:  
            l_sparse = torch.mean(torch.stack([p.mean() for p in sparse_probs]))
        else:
            l_sparse = torch.tensor(0.0, device=y_pred.device)

        if "astro_baseline_full" in model_outputs:
            astro_full = model_outputs["astro_baseline_full"]  # [B, L+H, C]
            truth_full = torch.cat([x_hist, y_true], dim=1)   # [B, L+H, C]

            truth_smooth = self.moving_avg(truth_full.permute(0, 2, 1)).permute(0, 2, 1)

            l_smooth = F.mse_loss(astro_full, truth_smooth)
            l_smooth_hist = torch.tensor(0.0, device=y_pred.device) 
            l_smooth_pred = torch.tensor(0.0, device=y_pred.device) 

            if not hasattr(self, '_first_call_printed'):
                self._first_call_printed = True
                
        else:
            x_hist_smooth = self.moving_avg(x_hist.permute(0, 2, 1)).permute(0, 2, 1)
            l_smooth_hist = F.mse_loss(b_ast_hist, x_hist_smooth)
            l_smooth_pred = torch.tensor(0.0, device=y_pred.device)
            if "prediction_astro" in model_outputs:
                y_ast_pred = model_outputs["prediction_astro"]
                y_true_smooth = self.moving_avg(y_true.permute(0, 2, 1)).permute(0, 2, 1)
                l_smooth_pred = F.mse_loss(y_ast_pred, y_true_smooth)
                
            l_smooth = l_smooth_hist + l_smooth_pred
            

        total_loss = l_pred + self.lambda_sparse * l_sparse + self.lambda_smooth * l_smooth

        loss_dict = {
            "total_loss": total_loss,
            "L_pred": l_pred,
            "L_FWCE_Rel": l_fwce_rel,
            "L_SDD": l_sdd,
            "L_sparse": l_sparse,
            "L_smooth": l_smooth,  
            "L_smooth_hist": l_smooth_hist, 
            "L_smooth_pred": l_smooth_pred,  
        }

        return total_loss, loss_dict
