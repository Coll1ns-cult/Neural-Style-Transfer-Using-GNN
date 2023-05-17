import torch 
import torch.nn as nn


class GATv2Layer(nn.Module):
    def __init__(self, in_features:int, out_features:int, 
    n_head: int, is_concat: bool = True,
    dropout: float  = 0.6,
    leaky_relu_slop: float = 0.2,
    share_weights:bool = False,
    dot_attention = False):
        super.__init__()
        self.is_concat = is_concat
        self.n_heads = n_heads
        self.share_weights = share_weights,
        self.dot_attention = dot_attention
    
        if is_concat:
            # hidden_dim = 
            assert out_features % n_head == 0
            self.hidden_dim = out_features//n_head
        else:
            self.hidden_dim = out_features
        
        self.Key = nn.Linear(in_features, self.hidden_dim * n_heads, bias = False)
        if share_weights:
            self.Query = self.Key
        else:
            self.Query = nn.Linear(in_features, self.hidden_dim*n_heads, bias = False)
        
        self.attn = nn.Linear(self.hidden_dim, 1, bias = False)
        self.activation = nn.LeakyRelu(negative_slop = leaky_relu_slop)
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(dropout)
    def forward(h, adj_mat):
        num_nodes = h.shape[0]
        key = self.Key(h).view(n_nodes, self.n_heads, self.hidden_dim)
        query = self.Query(h).view(n_nodes, self.n_heads, self.hidden_dim)

        key_repeat = key.repeat(n_nodes, 1, 1)
        query = query.repeat(n_nodes, dim = 0)
        g_sum = g_sum.view(n_nodes, n_nodes, self.n_heads, self.hidden_dim)

        scores = self.attn(self.activation(g_sum)).squeeze(-1)
        assert adj_mat.shape[0] == 1 or adj_mat.shape[0] == n_nodes
        assert adj_mat.shape[1] == 1 or adj_mat.shape[1] == n_nodes
        assert adj_mat.shape[2] == 1 or adj_mat.shape[2] == self.n_heads
        score = score.masked_fill(adj_mat == 0, float('-inf'))


        attention = self.dropout(self.softmax(score))


        attn_res = torch.einsum('ijh,jhf->ihf', a, g_r)

        if self.is_concat:
            return attn_res.reshape(n_nodes, self.n_heads * self.hidden_dim)
        else:
            return attn_res.mean(dim=1)





class GATv2(nn.Module):
    def __init__(self, in_features:int, out_features:int, 
    n_head: int, is_concat: bool = True,
    dropout: float  = 0.6,
    leaky_relu_slop: float = 0.2,
    share_weights:bool = False,
    dot_attention = False,
    num_layers: int):

    
    self.layers = nn.Sequential(*[GATv2Layer(in_features, out_features, n_head, dropout, leaky_relu_slop, share_weights, dot_attention) for _ in range(num_layers)])

    def forward(h, adj_mat):
        for layer in self.layers:
            h = layer(h, adj_mat)
        return h
