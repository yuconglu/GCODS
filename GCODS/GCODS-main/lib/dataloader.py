import torch
import numpy as np
import torch.utils.data
from lib.add_window import Add_Window_Horizon
from lib.load_dataset import load_st_dataset
from lib.normalization import NScaler, MinMax01Scaler, MinMax11Scaler, StandardScaler, ColumnMinMaxScaler
import pandas as pd
import os

class LazyWindowDataset(torch.utils.data.Dataset):
    def __init__(self, data, window, horizon, dataset_name=None, single=False):
        self.data = data
        self.window = window
        self.horizon = horizon
        self.dataset_name = dataset_name
        self.single = single
        
        length = len(data)
        end_index = length - horizon - window + 1
        self.length = end_index
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.single:
            X = self.data[idx:idx+self.window]
            Y = self.data[idx+self.window+self.horizon-1:idx+self.window+self.horizon]
        else:
            X = self.data[idx:idx+self.window]
            Y = self.data[idx+self.window:idx+self.window+self.horizon]
        
        return X, Y

def normalize_dataset(data, normalizer, column_wise=False, feature_wise = False):
    if normalizer == 'max01':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax01Scaler(minimum, maximum)
        data = scaler.transform(data)
    elif normalizer == 'max11':
        if column_wise:
            minimum = data.min(axis=0, keepdims=True)
            maximum = data.max(axis=0, keepdims=True)
        else:
            minimum = data.min()
            maximum = data.max()
        scaler = MinMax11Scaler(minimum, maximum)
        data = scaler.transform(data)
    elif normalizer == 'std':
        if column_wise:
            mean = data.mean(axis=0, keepdims=True)
            std = data.std(axis=0, keepdims=True)
        elif feature_wise:
            mean = data.mean(axis=(0,1), keepdims=True)
            std = data.std(axis=(0,1), keepdims=True)
        else:
            mean = data.mean()
            std = data.std()
        scaler = StandardScaler(mean, std)
        data = scaler.transform(data)
    elif normalizer == 'None':
        scaler = NScaler()
        data = scaler.transform(data)
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(data.min(axis=0), data.max(axis=0))
        data = scaler.transform(data)
    else:
        raise ValueError
    return data, scaler

def split_data_by_days(data, val_days, test_days, interval=60):
    T = int((24*60)/interval)
    test_data = data[-T*test_days:]
    val_data = data[-T*(test_days + val_days): -T*test_days]
    train_data = data[:-T*(test_days + val_days)]
    return train_data, val_data, test_data

def split_data_by_ratio(data, val_ratio, test_ratio):
    data_len = data.shape[0]
    test_data = data[-int(data_len*test_ratio):]
    val_data = data[-int(data_len*(test_ratio+val_ratio)):-int(data_len*test_ratio)]
    train_data = data[:-int(data_len*(test_ratio+val_ratio))]
    return train_data, val_data, test_data

def split_data_by_numbers(data, val_ratio, test_ratio):
    test_data = data[-int(test_ratio):]
    val_data = data[-int((test_ratio+val_ratio)):-int(test_ratio)]
    train_data = data[:-int((test_ratio+val_ratio))]
    return train_data, val_data, test_data

def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader


def get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
    data = load_st_dataset(args.dataset)

    base_data_path = args.data_dir
    if not os.path.isdir(base_data_path):
        raise ValueError(f"Data directory not found: {base_data_path}")

    time_coords_filename = 'time_coords_global_6H_2006_2018.npy'
    time_coords_path = os.path.join(base_data_path, time_coords_filename)
    
    try:
        time_coords = np.load(time_coords_path)
        if not np.issubdtype(time_coords.dtype, np.datetime64):
            time_coords = time_coords.astype('datetime64[ns]')
        time_index = pd.to_datetime(time_coords)
        
        if data.shape[0] != len(time_index):
            raise ValueError(f"Data time dimension ({data.shape[0]}) does not match time coordinates length ({len(time_index)}).")

    except FileNotFoundError:
        raise FileNotFoundError(f"Time coordinates file not found at: {time_coords_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading or processing time coordinates: {e}")

    train_mask = (time_index >= pd.Timestamp('2006-01-01')) & (time_index < pd.Timestamp('2016-01-01'))
    val_mask = (time_index >= pd.Timestamp('2016-01-01')) & (time_index < pd.Timestamp('2017-01-01'))
    test_mask = (time_index >= pd.Timestamp('2017-01-01')) & (time_index < pd.Timestamp('2019-01-01'))

    raw_data_train = data[train_mask]
    raw_data_val = data[val_mask]
    raw_data_test = data[test_mask]

    if normalizer == 'max01':
        if args.column_wise:
            minimum = raw_data_train.min(axis=0, keepdims=True)
            maximum = raw_data_train.max(axis=0, keepdims=True)
        else:
            minimum = raw_data_train.min()
            maximum = raw_data_train.max()
        scaler = MinMax01Scaler(minimum, maximum)
    elif normalizer == 'max11':
        if args.column_wise:
            minimum = raw_data_train.min(axis=0, keepdims=True)
            maximum = raw_data_train.max(axis=0, keepdims=True)
        else:
            minimum = raw_data_train.min()
            maximum = raw_data_train.max()
        scaler = MinMax11Scaler(minimum, maximum)
    elif normalizer == 'std':
        if args.column_wise:
            mean = raw_data_train.mean(axis=0, keepdims=True)
            std = raw_data_train.std(axis=0, keepdims=True)
        elif args.feature_wise:
            mean = raw_data_train.mean(axis=(0,1), keepdims=True)
            std = raw_data_train.std(axis=(0,1), keepdims=True)
        else:
            mean = raw_data_train.mean()
            std = raw_data_train.std()
        scaler = StandardScaler(mean, std)
    elif normalizer == 'None':
        scaler = NScaler()
    elif normalizer == 'cmax':
        scaler = ColumnMinMaxScaler(raw_data_train.min(axis=0), raw_data_train.max(axis=0))
    else:
        raise ValueError(f"Unknown normalizer type: {normalizer}")

    data_train = scaler.transform(raw_data_train)
    data_val = scaler.transform(raw_data_val)
    data_test = scaler.transform(raw_data_test)

    if data_train.shape[0] == 0 or data_test.shape[0] == 0:
        raise ValueError("Training or test set is empty after date-based splitting. Check data and time ranges.")

    raw_data_for_clim = load_st_dataset(args.dataset)
    raw_train_data_for_clim = raw_data_for_clim[train_mask]

    climatology_unnormalized = None
    if raw_train_data_for_clim.shape[0] > 0:
        climatology_unnormalized = raw_train_data_for_clim[:, :, :args.output_dim].mean(axis=0)
    else:
        climatology_unnormalized = np.zeros((args.num_nodes, args.output_dim))

    if args.time_dependence:
        time_index_train = time_index[train_mask]
        time_index_val = time_index[val_mask]
        time_index_test = time_index[test_mask]

        def get_time_feature(times_idx, num_nodes):
            feature = (times_idx.hour // 6).to_numpy()
            feature = feature / 3.0 
            feature_expanded = np.repeat(feature[:, np.newaxis], num_nodes, axis=1)
            feature_expanded = feature_expanded[..., np.newaxis]
            return feature_expanded

        if data_train.shape[0] > 0 :
            time_feat_train = get_time_feature(time_index_train, args.num_nodes)
            if data_train.shape[0] == time_feat_train.shape[0]:
                data_train = np.concatenate((data_train, time_feat_train), axis=-1)
        
        if data_val.shape[0] > 0:
            time_feat_val = get_time_feature(time_index_val, args.num_nodes)
            if data_val.shape[0] == time_feat_val.shape[0]:
                data_val = np.concatenate((data_val, time_feat_val), axis=-1)

        if data_test.shape[0] > 0:
            time_feat_test = get_time_feature(time_index_test, args.num_nodes)
            if data_test.shape[0] == time_feat_test.shape[0]:
                data_test = np.concatenate((data_test, time_feat_test), axis=-1)
        
    HORIZON_THRESHOLD = 12
    
    if args.horizon <= HORIZON_THRESHOLD:
        x_tra, y_tra = Add_Window_Horizon(data_train, args.lag, args.horizon, single, args.dataset)
        x_val, y_val = Add_Window_Horizon(data_val, args.lag, args.horizon, single, args.dataset)
        x_test, y_test = Add_Window_Horizon(data_test, args.lag, args.horizon, single, args.dataset)
        
        train_dataloader = data_loader(x_tra, y_tra, args.batch_size, shuffle=True, drop_last=True)
        if len(x_val) == 0:
            val_dataloader = None
        else:
            val_dataloader = data_loader(x_val, y_val, args.batch_size, shuffle=False, drop_last=True)
        test_dataloader = data_loader(x_test, y_test, args.batch_size, shuffle=False, drop_last=False)
    else:
        train_dataset = LazyWindowDataset(data_train, args.lag, args.horizon, args.dataset, single)
        val_dataset = LazyWindowDataset(data_val, args.lag, args.horizon, args.dataset, single) if data_val.shape[0] > 0 else None
        test_dataset = LazyWindowDataset(data_test, args.lag, args.horizon, args.dataset, single)
        
        def lazy_collate_fn(batch):
            cuda = True if torch.cuda.is_available() else False
            TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            
            X_list, Y_list = zip(*batch)
            
            X_np = np.stack([x for x in X_list])
            Y_np = np.stack([y for y in Y_list])
            
            X_batch = TensorFloat(X_np)
            Y_batch = TensorFloat(Y_np)
            
            return X_batch, Y_batch
        
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, 
            shuffle=True, drop_last=True, pin_memory=False,
            collate_fn=lazy_collate_fn
        )
        
        if val_dataset is not None and len(val_dataset) > 0:
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size,
                shuffle=False, drop_last=True, pin_memory=False,
                collate_fn=lazy_collate_fn
            )
        else:
            val_dataloader = None
        
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, pin_memory=False,
            collate_fn=lazy_collate_fn
        )
    
    return train_dataloader, val_dataloader, test_dataloader, scaler, climatology_unnormalized

if __name__ == '__main__':
    import argparse
    DATASET = 'SIGIR_electric'
    if DATASET == 'MetrLA':
        NODE_NUM = 207
    elif DATASET == 'BikeNYC':
        NODE_NUM = 128
    elif DATASET == 'SIGIR_solar':
        NODE_NUM = 137
    elif DATASET == 'SIGIR_electric':
        NODE_NUM = 321
    parser = argparse.ArgumentParser(description='PyTorch dataloader')
    parser.add_argument('--dataset', default=DATASET, type=str)
    parser.add_argument('--num_nodes', default=NODE_NUM, type=int)
    parser.add_argument('--processed_data_dir', default=None, type=str)
    parser.add_argument('--column_wise', action='store_true')
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--lag', default=12, type=int)
    parser.add_argument('--horizon', default=12, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--time_dependence', action='store_true')
    args = parser.parse_args()
    train_dataloader, val_dataloader, test_dataloader, scaler = get_dataloader(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True)