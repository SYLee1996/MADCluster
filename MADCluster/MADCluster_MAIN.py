import os

from MADCluster_UTILS import set_device, set_feature_size, he_init_normal, get_loader_segment, EarlyStopping
from MADCluster_MODEL import Network
from MADCluster_SOLVER import main_trainer

import argparse

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings("ignore")

# seed_everything()

def get_args_parser():
    parser = argparse.ArgumentParser('PyTorch Training', add_help=False)
    
    # Model parameters
    parser.add_argument('--MADCluster', action='store_true')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--patch_size', nargs='+', type=int, default=[3, 5, 7], help='List of patch_size')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--temperature', default=50.0, type=float)
    parser.add_argument('--nu', default=0.01, type=float)
    parser.add_argument('--objective', default='soft-boundary', type=str)
    
    # Optimizer parameters
    parser.add_argument('--smoothing_factor', default=0.01, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_lambda', default=0.9, type=float)
    parser.add_argument('--min_delta', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--init_method', default='he', type=str)
    parser.add_argument('--init_threshold', default=0.5, type=float)
    
    # Training parameters
    parser.add_argument('--dataset', default='MSL', type=str)
    parser.add_argument('--dataset_path', default='/home/sy.lee/OKESTRO/MADCluster/datasets/', type=str)
    parser.add_argument('--fname', default='M-2', type=str)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--patience', default=100, type=int)
    parser.add_argument('--device', default='0', type=str)
    parser.add_argument('--step', default=1, type=int)
    parser.add_argument('--mode', default='Train', type=str)

    return parser

def get_model_name(config):
    prefix = 'DCdetector_MADCluster' if config['MADCluster'] else 'DCdetector'

    key_params = [
        ('bs', config['batch_size']),
        ('ws', config['window_size']),
        ('ob', '_' + config['objective']),
        ('nu', f"{config['nu']:.3f}"),
        ('sf', f"{config['smoothing_factor']:.2f}"),
        ('lr', f"{config['lr']:.1e}"),
        ('wd', f"{config['weight_decay']:.1e}"),
        ('it', f"{config['init_threshold']:.2f}"),
        ('ep', config['num_epochs']),
        ('ds', config['dataset']),
        ('ps', '_'.join(map(str, config['patch_size']))),
        ('dm', config['d_model']),
        ('nh', config['n_heads']),
        ('el', config['e_layers']),
        ('k', config['k']),
        ('tp', config['temperature']),
    ]

    # convert parameter as string
    param_strs = [f"{k}{v}" for k, v in key_params]
    model_name = '_'.join([prefix] + param_strs)

    return model_name

def main(args):
    config = vars(args)

    assert config['mode'] in ('Train', 'Test'), "Mode must be either 'Train' or 'Test'."
    assert config['objective'] in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
    assert (0 < config['nu']) & (config['nu'] <= 1), "For hyperparameter nu, it must hold: 0 < nu <= 1."
    assert (0 < config['init_threshold']) & (config['init_threshold'] < 1), "For init_threshold, it must hold: 0 < init_threshold < 1."
    assert (0 < config['smoothing_factor']) & (config['smoothing_factor'] < 0.5), "For smoothing_factor, it must hold: 0 <= smoothing_factor <= 0.5."
    
    model_save_name = get_model_name(config)

    config['data_path'] = config['dataset_path'] + config['dataset']
    config['feature_size'] = set_feature_size(config['dataset'])
        
    print('------------ Options -------------')
    for k, v in sorted(config.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set results_dir relative to the script's directory
    results_dir = os.path.join(script_dir, 'RESULTS')
    if not os.path.exists(results_dir): 
        os.makedirs(results_dir)
        
    config['model_save_name'] = os.path.join(results_dir, model_save_name)
    print('model_save_name:', model_save_name)
    config['device'] = set_device(config['device'])
    
    # -------------------------------------------------------------------------------------------
    train_loader = get_loader_segment(config['data_path'], 
                                        batch_size=config['batch_size'], 
                                        window_size=config['window_size'],
                                        mode='train',
                                        step=config['step'],
                                        fname=config['fname'],
                                        dataset=config['dataset'])

    valid_loader = get_loader_segment(config['data_path'], 
                                    batch_size=config['batch_size'], 
                                    window_size=config['window_size'],
                                    mode='val',
                                    step=config['step'],
                                    fname=config['fname'],
                                    dataset=config['dataset'])

    test_loader = get_loader_segment(config['data_path'], 
                                    batch_size=config['batch_size'], 
                                    window_size=config['window_size'],
                                    mode='test',
                                    step=config['step'],
                                    fname=config['fname'],
                                    dataset=config['dataset'])
        
    # -------------------------------------------------------------------------------------------

    """Training the THOC model"""
    model = Network(config).to(config['device'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(config['device'])
                
    model.apply(he_init_normal)
    
    # initialize radius vector as much as number of clusters
    R = torch.zeros(1, device=config['device'])
    
    optimizer1 = torch.optim.Adam(model.get_model_params(), lr=config['lr'], betas=(0.9, 0.999), weight_decay=config['weight_decay'], amsgrad=True)
    if config['MADCluster']:
        # Create optimizer for 'thre' parameter
        optimizer2 = torch.optim.Adam(model.get_thre_param(), lr=config['lr'], amsgrad=True)
    else:
        optimizer2 = None
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer1, lr_lambda=lambda epoch: config['lr_lambda'] ** epoch)
    early_stopping_loss = EarlyStopping(patience=config['patience'], mode='min', min_delta=config['min_delta'])

    main_trainer(
        model=model, 
        scheduler=scheduler, 
        optimizer1=optimizer1, 
        optimizer2=optimizer2, 
        early_stopping_loss=early_stopping_loss, 
        config=config, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        test_loader=test_loader, 
        R=R)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
