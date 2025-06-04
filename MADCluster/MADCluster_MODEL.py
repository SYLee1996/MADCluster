import math

from math import sqrt
from itertools import chain
from einops import rearrange, reduce, repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.05):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)

class DAC_structure(nn.Module):
    def __init__(self, win_size, patch_size, channel, mask_flag=True, scale=None, attention_dropout=0.05, output_attention=False):
        super(DAC_structure, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.patch_size = patch_size
        self.channel = channel

    def forward(self, queries_patch_size, queries_patch_num, keys_patch_size, keys_patch_num, values, patch_index, attn_mask):

        # Patch-wise Representation
        B, L, H, E = queries_patch_size.shape #batch_size*channel, patch_num, n_head, d_model/n_head
        scale_patch_size = self.scale or 1. / sqrt(E)
        scores_patch_size = torch.einsum("blhe,bshe->bhls", queries_patch_size, keys_patch_size) #batch*ch, nheads, p_num, p_num   
        attn_patch_size = scale_patch_size * scores_patch_size
        series_patch_size = self.dropout(torch.softmax(attn_patch_size, dim=-1)) # B*D_model H N N
        
        # In-patch Representation
        B, L, H, E = queries_patch_num.shape #batch_size*channel, patch_size, n_head, d_model/n_head
        scale_patch_num = self.scale or 1. / sqrt(E)
        scores_patch_num = torch.einsum("blhe,bshe->bhls", queries_patch_num, keys_patch_num) #batch*ch, nheads, p_size, p_size 
        attn_patch_num = scale_patch_num * scores_patch_num
        series_patch_num = self.dropout(torch.softmax(attn_patch_num, dim=-1)) # B*D_model H S S 

        # Upsampling
        series_patch_size = repeat(series_patch_size, 'b l m n -> b l (m repeat_m) (n repeat_n)', repeat_m=self.patch_size[patch_index], repeat_n=self.patch_size[patch_index])    
        series_patch_num = series_patch_num.repeat(1,1,self.window_size//self.patch_size[patch_index],self.window_size//self.patch_size[patch_index]) 
        series_patch_size = reduce(series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
        series_patch_num = reduce(series_patch_num, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
        
        if self.output_attention:
            return series_patch_size, series_patch_num
        else:
            return (None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, patch_size, channel, n_heads, win_size, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.patch_size = patch_size
        self.channel = channel
        self.window_size = win_size
        self.n_heads = n_heads 
        
        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)      
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask):
        # patch_size
        B, L, M = x_patch_size.shape
        H = self.n_heads
        queries_patch_size, keys_patch_size = x_patch_size, x_patch_size
        queries_patch_size = self.patch_query_projection(queries_patch_size).view(B, L, H, -1) 
        keys_patch_size = self.patch_key_projection(keys_patch_size).view(B, L, H, -1) 

        # patch_num
        B, L, M = x_patch_num.shape
        queries_patch_num, keys_patch_num = x_patch_num, x_patch_num
        queries_patch_num = self.patch_query_projection(queries_patch_num).view(B, L, H, -1) 
        keys_patch_num = self.patch_key_projection(keys_patch_num).view(B, L, H, -1)
        
        # x_ori
        B, L, _ = x_ori.shape
        values = self.value_projection(x_ori).view(B, L, H, -1)
        
        series, prior = self.inner_attention(
            queries_patch_size, queries_patch_num,
            keys_patch_size, keys_patch_num,
            values, patch_index,
            attn_mask
        )
        
        return series, prior

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list

class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, dims, config):
        super(ClusteringLayer, self).__init__()
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, dims), requires_grad=True)
        
        # He initialization
        if config.get('init_method') == 'he':
            nn.init.kaiming_normal_(self.cluster_centers.data, mode='fan_in', nonlinearity='relu')

        # Xavier initialization
        elif config.get('init_method') == 'xavier':
            nn.init.xavier_normal_(self.cluster_centers.data)

        # Generic initialization (for other methods)
        elif config.get('init_method') == 'generic':
            std_dev = config.get('std_dev', 1.0)  # Default standard deviation
            mean = config.get('mean', 0.0)  # Default mean
            nn.init.normal_(self.cluster_centers.data, mean=mean, std=std_dev)

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        # cluster center shape: [n_clusters, dims]
        # Reshape x as [batch_size * seq_len, hidden_dim]
        x_flattened = x.reshape(-1, x.size(2))
        
        # Initialize an empty tensor to store similarities for each cluster
        similarities = torch.zeros(x_flattened.size(0), self.n_clusters, device=x.device)

		# Calculate cosine similarity for each cluster center

        for i in range(self.n_clusters):
            cluster_center = self.cluster_centers[i].unsqueeze(0)            
			
            similarity = F.cosine_similarity(x_flattened, cluster_center, dim=1)
			# bound similarity range as [0, 1]
            similarity = (similarity + 1) / 2
            similarities[:, i] = similarity
			
		# Reshape similarity as [batch_size, seq_len, n_clusters]
        similarities = similarities.view(x.size(0), x.size(1), self.n_clusters)
        
        return similarities

class Network(nn.Module):
    def __init__(self, config):
        super(Network, self).__init__()
        self.MADCluster = config['MADCluster']
        
        self.win_size = config['window_size']
        self.enc_in = config['feature_size']
        self.channel = config['feature_size']
        self.c_out = config['feature_size']
        self.patch_size = config['patch_size']
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.e_layers = config['e_layers']
        self.dropout = 0.1
        self.output_attention = True

        # Patching List  
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, self.d_model, self.dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, self.d_model, self.dropout))

        self.embedding_window_size = DataEmbedding(self.enc_in, self.d_model, self.dropout)
        
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(self.win_size, self.patch_size, self.channel, False, attention_dropout=self.dropout, output_attention=self.output_attention),
                    self.d_model, self.patch_size, self.channel, self.n_heads, self.win_size)for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)
        self.centroid = ClusteringLayer(1, self.d_model, config)
        self.threshold = nn.Parameter(torch.full((1,), config['init_threshold']), requires_grad=True)

    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []

        # Instance Normalization Operation
        x_ori = self.embedding_window_size(x)
        
        # Mutil-scale Patching Operation 
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
            
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)
        
        series_patch_mean = list(chain.from_iterable(series_patch_mean))
        prior_patch_mean = list(chain.from_iterable(prior_patch_mean))
                    
        if not self.MADCluster:
            return series_patch_mean, prior_patch_mean, x_ori, None, None, None, self.threshold
        
        # Compute the similarity for the clustering layer
        q = self.centroid(x_ori)

        weight = q**2 / q.sum(dim=(0, 1))
        p = weight / weight.sum(dim=2, keepdim=True)

        return series_patch_mean, prior_patch_mean, x_ori, p, q, self.centroid.cluster_centers, self.threshold

    def get_model_params(self):
        # Return model parameters excluding 'threshold'
        return [param for name, param in self.named_parameters() if name != 'threshold']

    def get_thre_param(self):
        # Return 'threshold' parameter
        return [self.threshold]
