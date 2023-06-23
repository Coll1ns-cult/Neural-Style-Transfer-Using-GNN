# Neural-Style-Transfer-Using-GNN

Replication of Learning Graph Neural Networks for Image Style Transfer (https://arxiv.org/abs/2207.11681)


Note: Paper has no implementation.
# Changes
Made
 - GATv2 instead of GAT, as graph in the network is bipartite graph in which dynamic attention is proven to be effective instead of static attention

 Planning
 - Implementing other types of graph construction to experiment which suits best for the problem. For example, using Threshold instead of KNN and etc
 
 
 To Do
 - Implementing Deformable Patch for style graph node construction.

# Results
In code.ipynb file
# Theoritical work
The idea was to utilize the fact that constructed style to content graph to be a bipartite graph in a way such that GATv2 performs much better compared to GAT in bipartite graphs, proven in the section Synthetic benchmark dictionary lookup in the paper of GATv2

