import os
import sys
file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(file_dir)
sys.path.append(file_dir)

import torch
import numpy as np
import torch.nn as nn
import argparse
import configparser
from datetime import datetime
from model.GCODS import GCODS as Network
from model.BasicTrainer import Trainer
from lib.TrainInits import init_seed
from lib.dataloader import get_dataloader
from lib.TrainInits import print_model_parameters
from lib.metrics import MAE_torch
import optuna

Mode = 'test'
DEBUG = False
DATASET = 'GIS'    
DEVICE = 'cuda'
MODEL = 'GCODS'

script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, '{}.conf'.format(DATASET))
print('Reading config file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file, encoding='utf-8')

def masked_mae_loss(scaler, mask_value):
    def loss(preds, labels):
        if scaler:
            preds = scaler.inverse_transform(preds)
            labels = scaler.inverse_transform(labels)
        mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
        return mae
    return loss

def run_training_job(trial=None):
    Mode = 'train'
    DEBUG = False
    DATASET = 'GIS'    
    DEVICE = 'cuda'
    MODEL = 'GCODS'

    config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}.conf'.format(DATASET))
    print(f'Reading config file: {config_file_path}')
    config = configparser.ConfigParser()
    config.read(config_file_path, encoding='utf-8')

    args = argparse.ArgumentParser(description='arguments')
    args.add_argument('--dataset', default=DATASET, type=str)
    args.add_argument('--mode', default=Mode, type=str)
    args.add_argument('--device', default=DEVICE, type=str)
    args.add_argument('--debug', default=DEBUG, type=eval)
    args.add_argument('--model', default=MODEL, type=str)
    args.add_argument('--cuda', default=True, type=bool)
    args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
    args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
    args.add_argument('--lag', default=config['data']['lag'], type=int)
    args.add_argument('--horizon', default=config['data']['horizon'], type=int)
    args.add_argument('--num_nodes', default=config['data']['num_nodes'], type=int)
    args.add_argument('--tod', default=config['data']['tod'], type=eval)
    args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
    args.add_argument('--column_wise', default=config['data']['column_wise'], type=eval)
    args.add_argument('--feature_wise', default=config['data']['feature_wise'], type=eval)
    args.add_argument('--data_dir', default=config['data'].get('data_dir', None), type=str)
    args.add_argument('--height', default=config['data'].getint('height', 32), type=int)
    args.add_argument('--width', default=config['data'].getint('width', 64), type=int)
    args.add_argument('--input_dim', default=config['model']['input_dim'], type=int)
    args.add_argument('--output_dim', default=config['model']['output_dim'], type=int)
    args.add_argument('--embed_dim', default=config['model']['embed_dim'], type=int) 
    args.add_argument('--hidden_dim', default=config['model']['hidden_dim'], type=int)
    args.add_argument('--alpha', default=config['model']['alpha'], type=float)
    args.add_argument('--time_dependence', default=config['model']['time_dependence'], type=eval)
    args.add_argument('--time_divided', default=config['model']['time_divided'], type=eval)
    args.add_argument('--model_type', default=config['model']['model_type'], type=str)
    args.add_argument('--use_cnn_encoder', default=config['model'].getboolean('use_cnn_encoder', False), type=eval)
    args.add_argument('--use_raw_encoder', default=config['model'].getboolean('use_raw_encoder', False), type=eval)
    args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
    args.add_argument('--seed', default=config['train']['seed'], type=int)
    args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    args.add_argument('--epochs', default=config['train']['epochs'], type=int)
    args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    args.add_argument('--teacher_forcing', default=False, type=bool)
    args.add_argument('--real_value', default=config['train']['real_value'], type=eval)
    args.add_argument('--warm_start', default=config['train'].getboolean('warm_start', False), type=eval)
    args.add_argument('--optimize_horizon', default=config['train'].get('optimize_horizon', 'average'), type=str)
    args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
    args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
    args.add_argument('--log_dir', default='./', type=str)
    args.add_argument('--log_step', default=config['log']['log_step'], type=int)
    args.add_argument('--plot', default=config['log']['plot'], type=eval)
    args.add_argument('--dropout_rate', type=float, default=config['train'].getfloat('dropout_rate', 0.1))
    args.add_argument('--r_drop_beta', type=float, default=config['train'].getfloat('r_drop_beta', 1.0))
    args.add_argument('--transformer_heads', type=int, default=config['model'].getint('transformer_heads', 4))
    args.add_argument('--transformer_dropout', type=float, default=config['model'].getfloat('transformer_dropout', 0.1))
    args.add_argument('--num_afno_blocks', type=int, default=config['model'].getint('num_afno_blocks', 8))
    args.add_argument('--use_dgcrn', default=config['model'].getboolean('use_dgcrn', False), type=eval)
    args.add_argument('--use_continuous_time', default=config['model'].getboolean('use_continuous_time', False), type=eval)
    args.add_argument('--time_step_hours', default=config['model'].getint('time_step_hours', 6), type=int)
    if config.has_section('dgcrn_params'):
        args.add_argument('--gcn_depth', default=config['dgcrn_params'].getint('gcn_depth', 2), type=int)
        args.add_argument('--dgcrn_dropout', default=config['dgcrn_params'].getfloat('dropout', 0.3), type=float)
        args.add_argument('--dgcrn_alpha', default=config['dgcrn_params'].getfloat('alpha', 0.05), type=float)
        args.add_argument('--dgcrn_beta', default=config['dgcrn_params'].getfloat('beta', 0.95), type=float)
        args.add_argument('--dgcrn_gamma', default=config['dgcrn_params'].getfloat('gamma', 0.95), type=float)
        args.add_argument('--node_dim', default=config['dgcrn_params'].getint('node_dim', 40), type=int)
        args.add_argument('--hyperGNN_dim', default=config['dgcrn_params'].getint('hyperGNN_dim', 16), type=int)
    else:
        args.add_argument('--gcn_depth', default=2, type=int)
        args.add_argument('--dgcrn_dropout', default=0.3, type=float)
        args.add_argument('--dgcrn_alpha', default=0.05, type=float)
        args.add_argument('--dgcrn_beta', default=0.95, type=float)
        args.add_argument('--dgcrn_gamma', default=0.95, type=float)
        args.add_argument('--node_dim', default=40, type=int)
        args.add_argument('--hyperGNN_dim', default=16, type=int)

    parsed_args = args.parse_known_args()[0]
    init_seed(parsed_args.seed)

    if torch.cuda.is_available() and parsed_args.cuda:
        parsed_args.device = 'cuda'
    else:
        parsed_args.device = 'cpu'

    if parsed_args.time_dependence:
        parsed_args.input_dim = parsed_args.input_dim + 1
        print(f"Time dependence enabled, model input_dim adjusted to: {parsed_args.input_dim}")

    train_loader, val_loader, test_loader, scaler, climatology_unnormalized = get_dataloader(parsed_args,
                                                                   normalizer=parsed_args.normalizer,
                                                                   tod=parsed_args.tod, dow=False,
                                                                   weather=False, single=False)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)

    from lib.load_dataset import get_adjacency_matrix
    edge_index = get_adjacency_matrix(parsed_args)

    model = Network(parsed_args, edge_index)
    model = model.to(parsed_args.device)

    if parsed_args.warm_start:
        model_path = os.path.join(script_dir, 'trained-best-model', f'best_model_{parsed_args.dataset}.pth')
        print(f"Checking warm start model file: {model_path}")
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=parsed_args.device, weights_only=False)
                model.load_state_dict(state_dict, strict=False)
                print(f"Warm start enabled. Successfully loaded pretrained model from: {model_path}")
            except Exception as e:
                print(f"Warning: Failed to load warm start model ({e}). Using random initialization.")
                for p in model.parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
                    else:
                        nn.init.uniform_(p)
        else:
            print(f"Warning: Warm start enabled but model not found at {model_path}. Using random initialization.")
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)
    else:
        print("Warm start disabled. Using random initialization...")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)
    
    print_model_parameters(model, only_num=False)

    if parsed_args.loss_func == 'mask_mae':
        loss = masked_mae_loss(scaler, mask_value=0.0)
    elif parsed_args.loss_func == 'mae':
        loss = torch.nn.L1Loss().to(parsed_args.device)
    elif parsed_args.loss_func == 'mse':
        loss = torch.nn.MSELoss().to(parsed_args.device)
    else:
        raise ValueError

    optimizer = torch.optim.Adam(params=model.parameters(), lr=parsed_args.lr_init, eps=1.0e-8,
                                 weight_decay=0.0005, amsgrad=False)
    lr_scheduler = None
    if parsed_args.lr_decay:
        print('Applying learning rate decay.')
        lr_decay_steps = [int(i) for i in list(parsed_args.lr_decay_step.split(','))]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                            milestones=lr_decay_steps,
                                                            gamma=parsed_args.lr_decay_rate)

    current_time = datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir_path_base = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', parsed_args.dataset)
    if trial:
        trial_log_dir = os.path.join(log_dir_path_base, f"trial_{trial.number}_{current_time}")
    else:
        trial_log_dir = os.path.join(log_dir_path_base, current_time)
    os.makedirs(trial_log_dir, exist_ok=True)
    parsed_args.log_dir = trial_log_dir

    trainer = Trainer(model, loss, optimizer, train_loader, val_loader, test_loader, scaler,
                      parsed_args, climatology_unnormalized, lr_scheduler=lr_scheduler)

    avg_val_weighted_rmse_for_trial = float('inf')
    value_for_optuna = float('inf')

    if parsed_args.mode == 'train':
        trainer.train()
        value_for_optuna = trainer.get_best_avg_val_weighted_rmse()
        best_rmse_epoch = trainer.get_best_avg_val_weighted_rmse_epoch()
        optimize_target = parsed_args.optimize_horizon if hasattr(parsed_args, 'optimize_horizon') else 'average'
        print(f"Training completed! Best RMSE ({optimize_target}): {value_for_optuna:.6f}")
        if best_rmse_epoch != -1:
            print(f"  Best performance at Epoch: {best_rmse_epoch}")
        else:
            print(f"  No best performance recorded")
        
        avg_overall_rmse = trainer.get_best_avg_val_weighted_rmse()
        avg_overall_rmse_epoch = trainer.get_best_avg_val_weighted_rmse_epoch()
        if avg_overall_rmse_epoch != -1:
            print(f"(Reference) Best validation Overall Average Weighted RMSE: {avg_overall_rmse:.6f} (Epoch {avg_overall_rmse_epoch})")

        if trial: 
            current_epoch_for_report = 1 
            trial.report(value_for_optuna, step=current_epoch_for_report) 
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    elif parsed_args.mode == 'test':
        logger = trainer.logger
        logger.info("--- Running in Test Mode ---")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, 'trained-best-model')
        model_path = os.path.join(model_dir, f"best_model_{parsed_args.dataset}.pth")
        
        if not os.path.exists(model_path):
            logger.error(f"Fatal: Best model not found at: {model_path}")
            logger.error("Please ensure a model has been trained before running in test mode.")
            return float('inf')

        logger.info(f"Loading best model for testing from: {model_path}")
        trainer.test(model, parsed_args, test_loader, scaler, logger, path_to_model_state=model_path)
        value_for_optuna = 0.0
    else:
        raise ValueError("Unknown mode: {}".format(parsed_args.mode))

    return value_for_optuna

if __name__ == '__main__':
    print("Run.py executed as main script.")

    script_dir_main = os.path.dirname(os.path.abspath(__file__))
    DATASET_main = 'GIS'
    config_file_main = os.path.join(script_dir_main, '{}.conf'.format(DATASET_main))
    print('Reading config file (main): %s' % (config_file_main))
    config_main = configparser.ConfigParser()
    config_main.read(config_file_main, encoding='utf-8')

    args_main_parser = argparse.ArgumentParser(description='arguments')
    args_main_parser.add_argument('--dataset', default=DATASET_main, type=str)
    args_main_parser.add_argument('--mode', default='train', type=str)
    args_main_parser.add_argument('--device', default='cuda', type=str)
    args_main_parser.add_argument('--debug', default=False, type=eval)
    args_main_parser.add_argument('--model', default='GCODS', type=str)
    args_main_parser.add_argument('--cuda', default=True, type=bool)
    args_main_parser.add_argument('--val_ratio', default=config_main['data']['val_ratio'], type=float)
    args_main_parser.add_argument('--test_ratio', default=config_main['data']['test_ratio'], type=float)
    args_main_parser.add_argument('--lag', default=config_main['data']['lag'], type=int)
    args_main_parser.add_argument('--horizon', default=config_main['data']['horizon'], type=int)
    args_main_parser.add_argument('--num_nodes', default=config_main['data']['num_nodes'], type=int)
    args_main_parser.add_argument('--tod', default=config_main['data']['tod'], type=eval)
    args_main_parser.add_argument('--normalizer', default=config_main['data']['normalizer'], type=str)
    args_main_parser.add_argument('--column_wise', default=config_main['data']['column_wise'], type=eval)
    args_main_parser.add_argument('--feature_wise', default=config_main['data']['feature_wise'], type=eval)
    args_main_parser.add_argument('--input_dim', default=config_main['model']['input_dim'], type=int)
    args_main_parser.add_argument('--output_dim', default=config_main['model']['output_dim'], type=int)
    args_main_parser.add_argument('--height', default=config_main['data'].getint('height', 32), type=int)
    args_main_parser.add_argument('--width', default=config_main['data'].getint('width', 64), type=int)
    args_main_parser.add_argument('--embed_dim', default=config_main['model']['embed_dim'], type=int)
    args_main_parser.add_argument('--hidden_dim', default=config_main['model']['hidden_dim'], type=int)
    args_main_parser.add_argument('--alpha', default=config_main['model']['alpha'], type=float)
    args_main_parser.add_argument('--time_dependence', default=config_main['model']['time_dependence'], type=eval)
    args_main_parser.add_argument('--time_divided', default=config_main['model']['time_divided'], type=eval)
    args_main_parser.add_argument('--model_type', default=config_main['model']['model_type'], type=str)
    args_main_parser.add_argument('--use_cnn_encoder', default=config_main['model'].getboolean('use_cnn_encoder', False), type=eval)
    args_main_parser.add_argument('--use_raw_encoder', default=config_main['model'].getboolean('use_raw_encoder', False), type=eval)
    args_main_parser.add_argument('--loss_func', default=config_main['train']['loss_func'], type=str)
    args_main_parser.add_argument('--seed', default=config_main['train']['seed'], type=int)
    args_main_parser.add_argument('--batch_size', default=config_main['train']['batch_size'], type=int)
    args_main_parser.add_argument('--epochs', default=config_main['train']['epochs'], type=int)
    args_main_parser.add_argument('--lr_init', default=config_main['train']['lr_init'], type=float)
    args_main_parser.add_argument('--lr_decay', default=config_main['train']['lr_decay'], type=eval)
    args_main_parser.add_argument('--lr_decay_rate', default=config_main['train']['lr_decay_rate'], type=float)
    args_main_parser.add_argument('--lr_decay_step', default=config_main['train']['lr_decay_step'], type=str)
    args_main_parser.add_argument('--early_stop', default=config_main['train']['early_stop'], type=eval)
    args_main_parser.add_argument('--early_stop_patience', default=config_main['train']['early_stop_patience'], type=int)
    args_main_parser.add_argument('--grad_norm', default=config_main['train']['grad_norm'], type=eval)
    args_main_parser.add_argument('--max_grad_norm', default=config_main['train']['max_grad_norm'], type=int)
    args_main_parser.add_argument('--teacher_forcing', default=False, type=bool)
    args_main_parser.add_argument('--real_value', default=config_main['train']['real_value'], type=eval)
    args_main_parser.add_argument('--warm_start', default=config_main['train'].getboolean('warm_start', False), type=eval)
    args_main_parser.add_argument('--optimize_horizon', default=config_main['train'].get('optimize_horizon', 'average'), type=str)
    args_main_parser.add_argument('--mae_thresh', default=config_main['test']['mae_thresh'], type=eval)
    args_main_parser.add_argument('--mape_thresh', default=config_main['test']['mape_thresh'], type=float)
    args_main_parser.add_argument('--log_dir', default='./', type=str)
    args_main_parser.add_argument('--log_step', default=config_main['log']['log_step'], type=int)
    args_main_parser.add_argument('--plot', default=config_main['log']['plot'], type=eval)
    args_main_parser.add_argument('--dropout_rate', type=float, default=config_main['train'].getfloat('dropout_rate', 0.1))
    args_main_parser.add_argument('--r_drop_beta', type=float, default=config_main['train'].getfloat('r_drop_beta', 1.0))
    args_main_parser.add_argument('--transformer_heads', type=int, default=config_main['model'].getint('transformer_heads', 4))
    args_main_parser.add_argument('--transformer_dropout', type=float, default=config_main['model'].getfloat('transformer_dropout', 0.1))
    args_main_parser.add_argument('--num_afno_blocks', type=int, default=config_main['model'].getint('num_afno_blocks', 8))
    args_main_parser.add_argument('--use_continuous_time', default=config_main['model'].getboolean('use_continuous_time', False), type=eval)
    args_main_parser.add_argument('--time_step_hours', default=config_main['model'].getint('time_step_hours', 6), type=int)
    args_main_parser.add_argument('--use_dgcrn', default=config_main['model'].getboolean('use_dgcrn', False), type=eval)
    if config_main.has_section('dgcrn_params'):
        args_main_parser.add_argument('--gcn_depth', default=config_main['dgcrn_params'].getint('gcn_depth', 2), type=int)
        args_main_parser.add_argument('--dgcrn_dropout', default=config_main['dgcrn_params'].getfloat('dropout', 0.3), type=float)
        args_main_parser.add_argument('--dgcrn_alpha', default=config_main['dgcrn_params'].getfloat('alpha', 0.05), type=float)
        args_main_parser.add_argument('--dgcrn_beta', default=config_main['dgcrn_params'].getfloat('beta', 0.95), type=float)
        args_main_parser.add_argument('--dgcrn_gamma', default=config_main['dgcrn_params'].getfloat('gamma', 0.95), type=float)
        args_main_parser.add_argument('--node_dim', default=config_main['dgcrn_params'].getint('node_dim', 40), type=int)
        args_main_parser.add_argument('--hyperGNN_dim', default=config_main['dgcrn_params'].getint('hyperGNN_dim', 16), type=int)
    
    args = args_main_parser.parse_args()

    print("Running a default training job as __main__...")
    default_rmse = run_training_job(trial=None)
    print(f"Default training job finished. Validation Avg Weighted RMSE: {default_rmse}")
