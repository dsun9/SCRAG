import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Epsilon = 1e-6

class VGAE(nn.Module):
    def __init__(self, num_user, num_assertion, input_dim, hidden1_dim, hidden2_dim):
        super(VGAE, self).__init__()
        self.input_dim = input_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.num_class = self.hidden2_dim
        self.base_gcn = GraphConvSparse(input_dim, hidden1_dim, activation=F.relu)
        self.gcn_mean = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x)
        self.gcn_logstddev = GraphConvSparse(hidden1_dim, hidden2_dim, activation=lambda x: x)

        self.num_user = num_user
        self.num_assertion = num_assertion
        self.user_nodes_mask = torch.zeros((self.num_user + self.num_assertion, self.hidden2_dim))#.to('cuda:0')
        self.user_nodes_mask[:self.num_user, :] = 1.0
        self.asser_nodes_mask = torch.zeros((self.num_user + self.num_assertion, self.hidden2_dim))#.to('cuda:0')
        self.asser_nodes_mask[self.num_user:, :] = 1.0

    def encode(self, adj, x):
        hidden = self.base_gcn(adj, x)
        mean = self.gcn_mean(adj, hidden)
        logstd = self.gcn_logstddev(adj, hidden)
        gaussian_noise = torch.randn(x.size(0), self.hidden2_dim, device=x.device)
        sampled_z = F.relu(gaussian_noise * torch.exp(logstd) + mean)
        # sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        return sampled_z, mean, logstd

    def decode(self, z):
        # Use this for bill
        self.user_nodes_mask = self.user_nodes_mask.to(z.device)
        self.asser_nodes_mask = self.asser_nodes_mask.to(z.device)
        u_b_m = torch.matmul(
             z * self.user_nodes_mask,
            (z * self.asser_nodes_mask).t()
        )
        return torch.sigmoid(u_b_m + u_b_m.t())

    def forward(self, adj, x):
        z = self.encode(adj, x)
        return self.decode(z)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation=F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, adj, inputs):
        x = inputs
        x = torch.mm(x, self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim) * 2 * init_range - init_range
    return nn.Parameter(initial)
