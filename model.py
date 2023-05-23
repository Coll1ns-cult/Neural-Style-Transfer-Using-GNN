
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.models as models
from gatv2 import GATv2Layer
from adain import AdaIn
vgg19 = models.vgg19(pretrained=True)

model_encoder = nn.Sequential()
# Remove the classification layer
#Reflection pad placements: afters: ReLU1_1, Relu1_2, relu, relu relu relu relu
i = 0
j = 1
first_sequential, _, _ = vgg19.children()

for layer in first_sequential.children():
    if torch.jit.isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv{}_{}'.format(j, i)
        layer.padding = (0, 0)
        ref_pad_name = 'reflection{}_{}'.format(j, i)
        model_encoder.add_module(ref_pad_name, nn.ReflectionPad2d(1))
    elif torch.jit.isinstance(layer, nn.MaxPool2d):
        j += 1
        i = 0

        name = 'pool{}'.format(j)
    elif torch.jit.isinstance(layer, nn.ReLU):

        name = 'relu{}_{}'.format(j, i)

    model_encoder.add_module(name, layer)
    if name == 'relu3_1':
        break

#now we have encoder: to call it: use model(input) to get encoded image

def patch2feat(image, patch_size, patch_stride):
    #dimension of image: NxCxHxW
    #patch dimension: K1xK2 = K
    #num_of_patches for image = P
    unfold = nn.Unfold(patch_size, patch_stride)
    patches = unfold(image) #image size: NxC*KxP
    return patches.transpose(1, 2) #dimensions: PxNxC*K
    # change: should do NxPxC*K so I will delete the first transpose


def feat2patch(feature, patch_size, output_size):
    if patch_size is tuple:
      k1, k2 = patch_size[0], patch_size[1]
    else:
      k1, k2 = patch_size, patch_size
    h, w = output_size[0], output_size[1]
    N, L, D = feature.size(0), feature.size(1), feature.size(2)
    C = D/(k1*k2)
    #after transpose NxPxC*K
    weight = torch.randn((int(C), int(C), int(k1), int(k2)), requires_grad = True)
    bias = torch.randn((1, int(C), int(h), int(w)), requires_grad = True)
    feature_convolved = feature.matmul(weight.view(weight.size(0), -1).t()).transpose(1, 2)
    #examples:
    fold = torch.nn.Fold(output_size, patch_size)
    feature_summed = fold(feature_convolved) + bias.expand(N, -1 ,-1, -1)
    return feature_summed 




class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.net = nn.Sequential(
                    nn.ReflectionPad2d(1), #4th block
                    nn.Conv2d(256, 128, (3, 3)),
                    nn.ReLU(inplace = True),
                    nn.ReflectionPad2d(1), #5th block
                    nn.Conv2d(128, 128, (3, 3)),
                    nn.ReLU(inplace = True),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d(1), #6th block
                    nn.Conv2d(128, 64, (3, 3)),
                    nn.ReLU(inplace = True ),
                    nn.ReflectionPad2d(1),#7th block
                    nn.Conv2d(64, 64, (3, 3)),
                    nn.ReLU(inplace = True),
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.ReflectionPad2d(1), #8th block
                    nn.Conv2d(64, 3, (3, 3)),
                    nn.ReLU(inplace = True),

)
    def forward(self, x):
        return self.net(x)
    

model_decoder = nn.Sequential()
i = 0
j = 1
first_sequential, _, _ = vgg19.children()

for layer in first_sequential.children():
    if torch.jit.isinstance(layer, nn.Conv2d):
        i += 1
        name = 'conv{}_{}'.format(j, i)
        # input_dim = layer.in_channels
        # output_dim = layer.out_channels
        # kernel_size = layer.kernel_size
        # layer1 = nn.Conv2d(input_dim, output_dim, kernel_size)
        layer.padding = (0, 0)
        ref_pad_name = 'reflection{}_{}'.format(j, i)
        model_decoder.add_module(ref_pad_name, nn.ReflectionPad2d(1))
    elif torch.jit.isinstance(layer, nn.MaxPool2d):
        j += 1
        i = 0
        nn.Upsample(scale_factor=2, mode='nearest')
        name = 'upsample{}'.format(j)
    elif torch.jit.isinstance(layer, nn.ReLU):

        name = 'relu{}_{}'.format(j, i)

    model_decoder.add_module(name, layer)
    if name == 'relu3_1':
        break



def knn(t, k, symm = True):
    k += 1
    if type(t) is tuple:
        content, style = t
        content = content / torch.norm(content, dim=2, keepdim=True) 
        style = style / torch.norm(style, dim=1, keepdim=True)
        nc, l, f = content.shape #nc, l, f is batch, node, and feature vector dimensions respectively
        if len(style.shape) == 2:
            p, f = style.shape #p is node, and f is feature dimension
            style = style.unsqueeze(0).expand(nc, p, f)
        else:
            _, p, f = style.shape #making dimensions of content to be 
        all = torch.cat((content, style), dim = 1)
        
        similarity = torch.matmul(all, all.transpose(1, 2)) # Ncx(l+p)x(l+p)
        similarity[:, l:, l:] = float('-inf')
        similarity[:, :l, :l] = float('-inf')



        _, indices = torch.topk(similarity, k, 1, True)
        indices = indices.transpose(1,2)

        total = l+p
        adj_matrix = torch.zeros(nc, total, total)
        adj_matrix = adj_matrix.scatter_(2, indices, 1)

        if not symm:
            adj_matrix[:, (l-1):, :l] = 0
        return adj_matrix
    else:
        content = t
        content = content / torch.norm(content, dim=2, keepdim=True) 

        nc, l, f = content.shape #nc, l, f is batch, node, and feature vector dimensions respectively
        similarity = torch.matmul(content, content.transpose(1, 2)) # Ncx(l)x(l)
        _, indices = torch.topk(similarity, k, 1, True)
        indices = indices.transpose(1,2)
        adj_matrix = torch.zeros(nc, l, l)
        adj_matrix = adj_matrix.scatter_(2, indices, 1)
        return adj_matrix

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)



class Net(nn.Module):
    def __init__(self, encoder:nn.Module
                 , decoder:nn.Module,
                #patches:
                patch_stride: int,
                patch_size: int,
                #value for KNN:
                k:int,
                #gnn inputs:
                n_heads: int, 
                in_features:int = 65536,
                is_concat: bool = True,
                dropout: float  = 0.6,
                leaky_relu_slop: float = 0.2,
                share_weights:bool = False,
                #whether to use pretrained decoder or not
                pre_trained_decoder = True,
                modulated_adain = False
                   ):
        super(self, Net).__init__()
        #encoder-decoder network
        self.encoder = encoder
        self.pre_trained_decoder = pre_trained_decoder
        if pre_trained_decoder:
            self.decoder = decoder #pretrained decoder
        else:
            self.decoder = decoder() #instance of decoder class that is created above

        #feat2patch, patch2feat opeartions
        self.patch_stide = patch_stride
        self.patch_size = patch_size

        #KNN
        self.k = k

        #gat layer class arguments
        self.in_fetures = in_features
        self.n_heads = n_heads
        self.dropout = dropout
        self.leaky_relu_slop = leaky_relu_slop
        self.share_weights = share_weights

        self.gatlayer1 = GATv2Layer(in_features, in_features, n_heads, is_concat, dropout , leaky_relu_slop, share_weights)
        self.gatlayer2 = GATv2Layer(in_features, in_features, n_heads, is_concat, dropout , leaky_relu_slop, share_weights)

        #adaptive instance normalization
        if self.modulated_adain:
            #initialize input dim with content features:
            self.adain = AdaIn(256, 128)

        

    def forward(self, style, content):
        #content and style should have same size because of gat
        encoded_style = self.encoder(style) #size of this: 1xCxHxW
        encoded_content = self.encoder(content) #size of this: NxCxHxW

        #patches of style and content
        style_patches = patch2feat(encoded_style)
        content_patches = patch2feat(encoded_content)


        #returning adjacency matrices for content to content, and content to style
        style_matrix = knn((content_patches, style_patches), self.k)

        updated_nodes = self.gatlayer1((content_patches, style_patches), style_matrix)

        updated_content_patches= updated_nodes[:, :content_patches[1]]

        content_matrix = knn(content_patches, self.k)
        final_content_patches = self.gatlayer2(updated_content_patches, content_matrix)

        #feat2patch:
        final_content = feat2patch(final_content_patches)

        #adaptive instance normalization:
        if self.modulated_adain:
            normalized_content = self.adain(final_content, encoded_style)
        else:
            normalized_content = adaptive_instance_normalization(final_content, encoded_style)

        output = self.decoder(normalized_content)

        return output
