# Neural-Style-Transfer-Using-GNN

Replication of Learning Graph Neural Networks for Image Style Transfer (https://arxiv.org/abs/2207.11681)


Note: Paper has no implementation.
# Changes
Made
 - GATv2 instead of GAT, as the graph in the network is a bipartite graph in which dynamic attention is proven to be effective instead of static attention

 Planning
 - Implementing other types of graph construction to experiment with which suits best for the problem. For example, using Threshold instead of KNN and etc
 
 
 To Do
 - Implementing Deformable Patch for style graph node construction.

# Results
The reason for this type of poor stylization and blurring is because of hyperparameters which were set. For example, the patch stride in paper is chosen to be 1, while in this implementation it is 7, otherwise, the number of constructed nodes will be too big, and memory issues will be faced. 
Content:
![content](https://github.com/Coll1ns-cult/Neural-Style-Transfer-Using-GNN/assets/70255599/e827c7e3-2699-4b06-a5c1-81919be608ad)

Style:
![style1](https://github.com/Coll1ns-cult/Neural-Style-Transfer-Using-GNN/assets/70255599/38052067-2982-4f1f-9eab-f535860e2506)

Result:
<img width="186" alt="Screen Shot 2023-06-23 at 19 09 56" src="https://github.com/Coll1ns-cult/Neural-Style-Transfer-Using-GNN/assets/70255599/2cf3d98a-69be-4f6f-b501-977372b6c576">




# Theoritical work
The idea was to utilize the fact that constructed style to content graph to be a bipartite graph in a way such that GATv2 performs much better compared to GAT in bipartite graphs, proven in the section Synthetic benchmark dictionary lookup in the paper of GATv2

