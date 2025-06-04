import copy
import numpy as np

from tqdm import tqdm

from sklearn import metrics
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from vus.metrics import get_metrics
    
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def custom_loss_function(p, q, trhe, config, epsilon=1e-6, smoothing_factor=0.01):
    smoothing_factor = config['smoothing_factor']
    
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)

    # kl_loss shape: [batch_size, sequence_size, number_of_clusters]
    kl_div = p * torch.log(p / q)
    
    p = p * (1 - smoothing_factor) + (1 - p) * smoothing_factor
    term1 = p * torch.log(torch.clamp((1 - trhe**(1-trhe)) / (1 - trhe) * (q - trhe) + trhe**(1 - trhe), max=1e+6))
    term2 = (1 - p) * torch.log(torch.clamp(q**(1 - trhe), max=1e+6))
    
    # trhe_loss shape: [batch_size, sequence_size, number_of_clusters]
    trhe_loss = -(term1 + term2)
    loss = kl_div + trhe_loss
    return loss

def mapping_loss_function(z, c, R, config):
    # z, c shape:                      [batch_size, sequence_size, features_size] & [number_of_clusters, features_size]
    # z_expanded, c_expanded shape:    [batch_size, sequence_size, number_of_clusters, features_size]
    
    z_expanded = z.unsqueeze(2).expand(-1, -1, c.size(0), -1)
    c_expanded = c.unsqueeze(0).unsqueeze(0).expand(z.size(0), z.size(1), -1, -1)
    
    # Calculate the squared distance
    dist = torch.sum((z_expanded - c_expanded) ** 2, dim=-1)
        
    if config['objective'] == 'soft-boundary':
        R_expanded = R.unsqueeze(0).unsqueeze(0).expand(z.size(0), z.size(1), -1)
        scores = dist - R_expanded ** 2 
        mapping_loss = R_expanded ** 2 + (1 / config['nu']) * torch.max(torch.zeros_like(scores), scores)
    else:
        scores = dist
        mapping_loss = dist
        
    # mapping_loss shape: [batch_size, sequence_size, number_of_clusters]
    return dist, scores, mapping_loss

def get_radius(dist: torch.Tensor, nu: float):
    return np.array([np.quantile(np.sqrt(dist[:, :, i].clone().data.cpu().numpy()), 1 - nu) for i in range(dist.shape[2])])

def training(config, data_loader, model, optimizer1, optimizer2, epoch_num, R):
    model.train()
    pbar = tqdm(data_loader, total=len(data_loader), desc=f"Train Epoch {epoch_num}")
    
    avg_dc_loss = 0
    avg_mapping_loss = 0
    avg_cluster_loss = 0
    avg_total_loss = 0
    all_dists = []
    
    for i, (x, _) in enumerate(pbar):
        x = x.to(config['device']) 
        
        if config['MADCluster'] == True:
            optimizer2.zero_grad()
            
        # 손실 계산
        if config['MADCluster'] is not True:
            optimizer1.zero_grad()
            
            series, prior, x_latent, _, _, _, threshold = model(x)
            
            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                config['window_size'])).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        config['window_size'])).detach(),
                                series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            config['window_size'])),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])))))
            # print('\n')
            # print(series_loss)
            # print(prior_loss)
            # print('\n')
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            
            loss = prior_loss - series_loss
            
            mapping_loss = torch.tensor(0.0).to(config['device'])
            cluster_loss = torch.tensor(0.0).to(config['device'])
            dist = torch.tensor([0.0]).to(config['device'])
            loss_sum = torch.tensor(0.0).to(config['device'])
        else:
            optimizer1.zero_grad()
            
            series, prior, x_latent, p, q, c, threshold = model(x)
            
            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                config['window_size'])).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        config['window_size'])).detach(),
                                series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            config['window_size'])),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            
            loss = prior_loss - series_loss
            
            cluster_loss = custom_loss_function(p, q, threshold, config)
            dist, scores, mapping_loss = mapping_loss_function(x_latent, c, R, config)
            
            # 1. summation of mapping_loss and cluster loss
            sum_loss = mapping_loss + cluster_loss
            
            # 2. summation about dimensionality of cluster
            cluster_sum = torch.sum(sum_loss, dim=[1, 2])
            
            # 3. mean of all of samples
            loss_sum = torch.log(torch.mean(cluster_sum) + 1e-8)  # scalar
            
        losses = loss + loss_sum
        
        # Minimax strategy
        losses.backward()

        optimizer1.step()

        if config['MADCluster'] == True:
            optimizer2.step()
        
        # 평균 손실 업데이트
        avg_dc_loss = (avg_dc_loss * i + loss.mean().item()) / (i + 1)
        avg_mapping_loss = (avg_mapping_loss * i + mapping_loss.mean().item()) / (i + 1)
        avg_cluster_loss = (avg_cluster_loss * i + cluster_loss.sum().item()) / (i + 1)
        avg_total_loss = avg_dc_loss + avg_mapping_loss + avg_cluster_loss
                    
        # 진행 상태 업데이트
        pbar.set_postfix({
            'Thre': ', '.join([f'{t:.5f}' for t in threshold.tolist()])
        })
        
        # append the dist by each cluster
        all_dists.append(dist.detach().cpu())
        
        # concatenate all_dists
    all_dists = torch.cat(all_dists, dim=0)

    return model, threshold, avg_dc_loss, avg_mapping_loss, avg_cluster_loss, avg_total_loss, all_dists

def validation(config, data_loader, model, R):
    loss_dict = {
        'dc_loss': 0,
        'mapping_loss': 0,
        'cluster_loss': 0,
    }
    model.eval()
    pbar = tqdm(data_loader, total=len(data_loader), desc="Validation")  # Initialize the progress bar
    all_dists = []
    
    with torch.no_grad():
        for i, (x, _) in enumerate(pbar):
            x = x.to(config['device']) 
            
            if config['MADCluster'] is not True:
                series, prior, x_latent, _, _, _, threshold = model(x)
                
                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            config['window_size'])).detach(),
                                    series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                config['window_size'])),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        config['window_size'])))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                
                loss = prior_loss - series_loss
                
                mapping_loss = torch.tensor(0.0).to(config['device'])
                cluster_loss = torch.tensor(0.0).to(config['device'])
                dist = torch.tensor([0.0]).to(config['device'])
            else:            
                series, prior, x_latent, p, q, c, threshold = model(x)
                
                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])).detach())) + torch.mean(
                        my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                            config['window_size'])).detach(),
                                    series[u])))
                    prior_loss += (torch.mean(my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                config['window_size'])),
                        series[u].detach())) + torch.mean(
                        my_kl_loss(series[u].detach(), (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                        config['window_size'])))))

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                
                loss = prior_loss - series_loss
                
                cluster_loss = custom_loss_function(p, q, threshold, config)
                dist, scores, mapping_loss = mapping_loss_function(x_latent, c, R, config)
                
            loss_dict['dc_loss'] += loss.mean().item()
            loss_dict['mapping_loss'] += mapping_loss.mean()             
            loss_dict['cluster_loss'] += cluster_loss.sum() 
            
            all_dists.append(dist.detach().cpu())
        
    loss_dict['dc_loss'] /= len(data_loader)
    loss_dict['mapping_loss'] /= len(data_loader)
    loss_dict['cluster_loss'] /= len(data_loader)

    all_dists = torch.cat(all_dists, dim=0)    
    
    return loss_dict, all_dists

def testing(config, test_loader, model):
    real_y, test_results = [], []

    model.eval()
    with torch.no_grad():
        for i, (x,y) in enumerate(test_loader):
            x = x.to(config['device']) 
            y = y.to(config['device'])     
            
            if config['MADCluster'] is not True:
                series, prior, x_latent, _, _, _, threshold = model(x)
                
                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])).detach()) * config['temperature']
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])),
                            series[u].detach()) * config['temperature']
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])).detach()) * config['temperature']
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])),
                            series[u].detach()) * config['temperature']

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)                
                metric = torch.mean(metric, dim=1) 
                
                Anomaly_score = metric
            else:            
                series, prior, x_latent, p, q, c, threshold = model(x)
                
                # calculate Association discrepancy
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    if u == 0:
                        series_loss = my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])).detach()) * config['temperature']
                        prior_loss = my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])),
                            series[u].detach()) * config['temperature']
                    else:
                        series_loss += my_kl_loss(series[u], (
                                prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])).detach()) * config['temperature']
                        prior_loss += my_kl_loss(
                            (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                    config['window_size'])),
                            series[u].detach()) * config['temperature']

                metric = torch.softmax((-series_loss - prior_loss), dim=-1)                
                metric = torch.mean(metric, dim=1) 
                
                cluster_loss = custom_loss_function(p, q, threshold, config)
                sum_scores = cluster_loss                
                cluster_sum = torch.mean(sum_scores, dim=[1, 2])

                Anomaly_score = metric * cluster_sum

            y_2d = (y.sum(dim=1, keepdim=True) > 0).float()
            y_1d = y_2d.squeeze(1)
            
            test_results.append(Anomaly_score)
            real_y.append(y_1d)
    
    y_pred_test = torch.cat(test_results).detach().cpu().numpy()
    y_real = torch.cat(real_y).detach().cpu().numpy().squeeze()
    
    prec, rec, f1, aupr, roc_auc, threshold = calculate_metrics_per_dataset(y_real, y_pred_test)
    
    metric = {'R_AUC_ROC':None, 'R_AUC_PR':None, 'VUS_ROC':None, 'VUS_PR':None, 'Affiliation_Precision':None, 'Affiliation_Recall':None}
    window_pred = (y_pred_test > threshold)

    for m in metric.keys():
        metric[m] = get_metrics(window_pred.astype(int), y_real, metric=m, slidingWindow=config['window_size'])
    
    print("\nEvaluation Metrics:")
    print("-" * 60)
    print(f"{'Precision':<25}: {prec:.5f}")
    print(f"{'Recall':<25}: {rec:.5f}")
    print(f"{'F1-score':<25}: {f1:.5f}")
    print(f"{'AU-PR':<25}: {aupr:.5f}")
    print(f"{'AU-ROC':<25}: {roc_auc:.5f}")
    print(f"{'R_AUC_ROC':<25}: {metric['R_AUC_ROC']:.5f}")
    print(f"{'R_AUC_PR':<25}: {metric['R_AUC_PR']:.5f}")
    print(f"{'VUS_ROC':<25}: {metric['VUS_ROC']:.5f}")
    print(f"{'VUS_PR':<25}: {metric['VUS_PR']:.5f}")
    print(f"{'Affiliation_Precision':<25}: {metric['Affiliation_Precision']:.5f}")
    print(f"{'Affiliation_Recall':<25}: {metric['Affiliation_Recall']:.5f}")
    print("-" * 60)

    print()
    print()
    return prec, rec, f1, aupr, roc_auc, metric['R_AUC_ROC'], metric['R_AUC_PR'], metric['VUS_ROC'], metric['VUS_PR'], metric['Affiliation_Precision'], metric['Affiliation_Recall']
    
def calculate_metrics_per_dataset(true_labels, predictions):
    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)

    f1_scores = np.zeros_like(precision)  
    for i in range(len(precision)):
        if precision[i] + recall[i] > 0:
            f1_scores[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        else:
            f1_scores[i] = 0.0 
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    
    roc_auc = roc_auc_score(true_labels, predictions)
    pr_auc = average_precision_score(true_labels, predictions)

    first_prec = precision[best_f1_index] if len(precision) > 0 else 0.0
    first_rec = recall[best_f1_index] if len(recall) > 0 else 0.0
    
    best_threshold = thresholds[best_f1_index]
    
    return first_prec, first_rec, best_f1, pr_auc, roc_auc, best_threshold

def main_trainer(
    model, 
    scheduler, 
    optimizer1, 
    optimizer2, 
    early_stopping_loss, 
    config, 
    train_loader, 
    valid_loader, 
    test_loader, 
    R):

    results = {
        'TRAIN_dc_loss': [], 'TRAIN_total_loss': [], 'TRAIN_mapping_loss': [], 'TRAIN_cluster_loss': [],
        'VALID_dc_loss': [], 'VALID_total_loss': [], 'VALID_mapping_loss': [], 'VALID_cluster_loss': [],
        'radius': [], 'threshold': []
    }
    
    if config['mode'] == 'Train':
        best_loss = np.inf
        
        for epoch in range(config['num_epochs']):
            epoch_num = epoch + 1
            
            model, threshold, avg_dc_loss, avg_mapping_loss, avg_cluster_loss, avg_total_loss, train_dists = training(config=config, 
                                                                                                                                        data_loader=train_loader, 
                                                                                                                                        model=model, 
                                                                                                                                        optimizer1=optimizer1, 
                                                                                                                                        optimizer2=optimizer2, 
                                                                                                                                        epoch_num=epoch_num,
                                                                                                                                        R=R)
            scheduler.step()

            VALID_loss_dict, valid_dists = validation(config=config,
                                                      data_loader=valid_loader,
                                                      model=model,
                                                      R=R)
            valid_total_loss = VALID_loss_dict['dc_loss'] + \
                               VALID_loss_dict['mapping_loss'] + \
                               VALID_loss_dict['cluster_loss']
                               
            # calculate to new radius by each cluster
            all_dists = torch.cat([train_dists, valid_dists], dim=0)
            
            if config['MADCluster'] == True: 
                R = torch.tensor(get_radius(all_dists, config['nu']), device=config['device'])
            
            results['TRAIN_dc_loss'].append(avg_dc_loss)
            results['TRAIN_total_loss'].append(avg_total_loss)
            results['TRAIN_mapping_loss'].append(avg_mapping_loss)
            results['TRAIN_cluster_loss'].append(avg_cluster_loss)
            
            results['VALID_dc_loss'].append(VALID_loss_dict['dc_loss'])
            results['VALID_total_loss'].append(valid_total_loss)
            results['VALID_mapping_loss'].append(VALID_loss_dict['mapping_loss'])
            results['VALID_cluster_loss'].append(VALID_loss_dict['cluster_loss'])
            results['radius'].append(R.tolist())
            results['threshold'].append(threshold.tolist())  
            
            print(f'\nEpoch {epoch_num} Summary:')
            
            print(f'Train - Total Loss: {avg_total_loss:.5f}, '
                  f'TRAIN_dc Loss: {avg_dc_loss:.5f}, '
                  f'Mapping Loss: {avg_mapping_loss:.5f}, '
                  f'Clustering Loss: {avg_cluster_loss:.5f}')
            
            print(f'Valid - Total Loss: {valid_total_loss:.5f}, '
                  f"Valid_dc Loss: {VALID_loss_dict['dc_loss']:.5f}, "
                  f"Mapping Loss: {VALID_loss_dict['mapping_loss']:.5f}, "
                  f"Clustering Loss: {VALID_loss_dict['cluster_loss']:.5f}")
            
            print(f'Threshold: {[f"{t:.5f}" for t in threshold.tolist()]}, Radius: {[f"{r:.5f}" for r in R.tolist()]}\n')

            if valid_total_loss < best_loss:
                improvement = best_loss - valid_total_loss
                if improvement >= config['min_delta']:
                    best_loss = valid_total_loss
                    best_idx = epoch_num
                    model_state_dict = model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict()
                    best_model_wts = copy.deepcopy(model_state_dict)
                    
                    torch.save({'R': R, 'net_dict': best_model_wts}, f'{config["model_save_name"]}.pt')
                    
                    load_best_model_wts = torch.load(f'{config["model_save_name"]}.pt')

                    if torch.cuda.device_count() > 1:
                        model.module.load_state_dict(load_best_model_wts['net_dict'])
                    else:
                        model.load_state_dict(load_best_model_wts['net_dict'])
                    
                    R = load_best_model_wts['R'].to(config['device'])
                    
                    print(f'==> best model saved {best_idx} epoch / loss : {valid_total_loss:.8f}')
                else:
                    print(f'Loss improved by {improvement:.8f}, but less than min_delta ({config["min_delta"]}). Not saving model.')
            else:
                print(f'Loss did not improve. Current: {valid_total_loss:.8f}, Best: {best_loss:.8f}')

            if early_stopping_loss.step(torch.tensor(valid_total_loss)):
                print("Early stopping")
                break
            
            # testing(config, train_loader, thre_loader, model, R)
            prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall = testing(config, test_loader, model)
        
        prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall = testing(config, test_loader, model)
        
        process_results_and_save(config, config['dataset'], prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall)

    elif config['mode'] == 'Test':
        prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall = testing(config, test_loader, model)
        
        process_results_and_save(config, config['dataset'], prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall)
        
        
def process_results_and_save(config, dataset_name, prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall):
    """
    Helper function to save results to CSV based on the dataset name.
    """
    fname = 'All-data' if dataset_name in ['PSM', 'SWaT', 'WADI'] else config['fname']
    save_results_to_csv(config, fname, prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall)

        
import os 
import pandas as pd

def save_results_to_csv(config, fname, prec, rec, f1, aupr, roc_auc, R_AUC_ROC, R_AUC_PR, VUS_ROC, VUS_PR, Affiliation_Precision, Affiliation_Recall):
    # Define the output CSV file path
    results_file = f"{config['model_save_name']}_results.csv"
    
    # Create a DataFrame with the results for the current epoch
    results = {
        'fname': fname,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'aupr': aupr,
        'roc_auc': roc_auc,
        'R_AUC_ROC': R_AUC_ROC, 
        'R_AUC_PR': R_AUC_PR, 
        'VUS_ROC':VUS_ROC,
        'VUS_PR':VUS_PR, 
        'Affiliation_Precision':Affiliation_Precision,
        'Affiliation_Recall': Affiliation_Recall
    }
    
    df = pd.DataFrame([results])

    # Append the results to the CSV file
    if not os.path.exists(results_file):
        df.to_csv(results_file, index=False)
    else:
        df.to_csv(results_file, mode='a', header=False, index=False)
