
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class Dataset_QiantangTidal(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='cq1_processed.csv', scale=True):

        if size is None:
            self.seq_len = 100
            self.pred_len = 10
        else:
            self.seq_len = size[0]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        self.scaler_dynamic = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path),
                             parse_dates=['Event_Date'])

        df_raw['Year'] = df_raw['Event_Date'].dt.year
        df_raw['Month'] = df_raw['Event_Date'].dt.month
        dynamic_features = ['Low_Level', 'High_Level', 'Delta_t_Rise_hr', 'Delta_t_Cycle_hr']
        periodic_features = ['Day_of_Year', 'Lunar_Day', 'Month']
        year_feature = ['Year']

        data_dynamic = df_raw[dynamic_features]
        data_periodic = df_raw[periodic_features]
        data_year = df_raw[year_feature]

        num_train = int(len(df_raw) * 0.7)
        num_test = len(df_raw) - num_train

        # 'val' set will use the same data as the 'test' set.
        border1s = [0, num_train - self.seq_len, num_train - self.seq_len]
        border2s = [num_train, len(df_raw), len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
      
        if self.scale:
            train_dynamic_data = data_dynamic.iloc[border1s[0]:border2s[0]].values
            self.scaler_dynamic.fit(train_dynamic_data)
            data_dynamic_scaled = self.scaler_dynamic.transform(data_dynamic.values)
        else:
            data_dynamic_scaled = data_dynamic.values


        year_cyclic = data_year.values % 4 
        data_year_scaled = year_cyclic + 1 



        data_periodic_raw = data_periodic.values


        self.data_dynamic = data_dynamic_scaled[border1:border2]
        self.data_periodic = data_periodic_raw[border1:border2]
        self.data_year = data_year_scaled[border1:border2]

    def __len__(self):
        return len(self.data_dynamic) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        x_hist_dynamic = self.data_dynamic[s_begin:s_end]
        x_hist_periodic = self.data_periodic[s_begin:s_end]
        x_hist_year = self.data_year[s_begin:s_end]

        y_true_dynamic = self.data_dynamic[r_begin:r_end]
        y_fut_periodic = self.data_periodic[r_begin:r_end]
        y_fut_year = self.data_year[r_begin:r_end]

        return (torch.from_numpy(x_hist_dynamic).float(), torch.from_numpy(x_hist_periodic).float(),
                torch.from_numpy(x_hist_year).float(), torch.from_numpy(y_true_dynamic).float(),
                torch.from_numpy(y_fut_periodic).float(), torch.from_numpy(y_fut_year).float())

    def inverse_transform_dynamic(self, data):
        if self.scale:
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            return self.scaler_dynamic.inverse_transform(data)
        return data
