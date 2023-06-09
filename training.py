import torch
import torch.nn as nn
import torch.optim as Optimizer
import torch.nn.functional as F
import tqdm
import deeplake
from torchvision import datasets, transforms, models
# from torch.utils.tensorboard import SummaryWriter
# from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

from model import Net

unloader = transforms.ToPILImage()  # reconvert into PIL image



ds_c = deeplake.load('hub://activeloop/coco-train') # Deep Lake Dataset
ds_c.visualize()
ds_s = deeplake.load('hub://activeloop/wiki-art')
ds_s.visualize()

tform = transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
])

# content Dataloader
c_loader= ds_c.pytorch(batch_size = 4, transform = {'images': tform}, shuffle = False)
c_iter = iter(c_loader)

# style dataloader
s_loader= ds_s.pytorch(batch_size = 1, transform = {'images': tform, 'labels': None}, shuffle = False)
s_iter = iter(s_loader)

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
#Look this part again, define parameter to be optimized. 
# optimizer = Optimizer.Adam(params = model, lr = 1e-4, weight_decay=5e-5)

content_layers = ['relu4_1']
style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1']

def train(model: nn.Module,
        #   optimizer: Optimizer,
          pretrained_vgg: nn.Module,
          content_iter,
          style_iter,
        #   val_dataloader:torch.utils.data.DataLoader,
          device:torch.cuda.device,
          max_iter,
          loss_factor:float = 10
        #   epoch_count:int,
        #   tb_writer:torch.utils.tensorboard
          ):
    '''ToDo: add logs to see overfitting and etc.
           : Add Normalization over statistics of dataset which VGG is trained over'''
    optimizer = Optimizer.Adam(params = model.parameters, lr = 1e-4, weight_decay=5e-5)
    model.to(device)
    model.train()
    for i in tqdm(range(max_iter)):
        content = next(content_iter)
        style = next(style_iter)
        running_loss = 0
        content.todevice(device)
        style.todevice(device)
        output = model(content, style) #stylized content image

        vgg = nn.Sequential()

        content_losses = []
        style_losses = []
        first_sequential, _, _ = vgg19.children()
        i = 0
        j = 1
        for layer in first_sequential.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv{}_{}'.format(j, i)
            elif isinstance(layer, nn.MaxPool2d):
                j += 1
                name = 'pool{}'.format(j)
            elif isinstance(layer, nn.ReLU):
                name = 'relu{}_{}'.format(j, i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'batchnorm{}_{}'.format(j, i)

            vgg.add_module(name, layer)

            if name in content_layers:
                output = vgg(output)
                content_loss = F.mse_loss(output, content)
                content_losses.append(content_loss)
            if name in style_layers:
                output = vgg(output)
                output_mean, output_std =  torch.mean(output, dim = 0), torch.std(output, dim = 0)
                style_output = vgg(style)
                style_output_mean, style_output_std = torch.mean(style_output, dim = 0), torch.std(style_output, dim = 0)
                style_std_loss = F.mse_loss(output_std, style_output_std)
                style_mean_loss = F.mse_loss(output_mean, style_output_mean)
                style_loss = style_std_loss + style_mean_loss
                style_losses.append(style_loss)



        total_loss = style_loss*loss_factor + content_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss +=total_loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 100 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_count * len(training_dataloader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    print("###########################################")
    # model.eval()
    # for i, data in enumerate(val_dataloader):
    #     content, style = data
    #     content.todevice(device)
    #     style.todevice(device)
    #     output = model(content, style)
    #     if i % 5000 == 4999:
    #         plt.figure()
    #         imshow(output, title = 'stylized image of iter {}'.format(i))

    return 







# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

# def train(epochs:int,
#         one_epoch_trainer,
#         model: nn.Module,
#         #   optimizer: Optimizer,
#         pretrained_vgg: nn.Module,
#         training_dataloader: torch.utils.data.DataLoader, 
#         val_dataloader:torch.utils.data.DataLoader,
#         device:torch.cuda.device,
#         loss_factor:float,
#         loss_function:nn.Module,
#         epoch_count:int,
#         tb_writer:torch.utils.tensorboard
#         ):
#     for epoch in range(epochs):



model = Net()
ds_c = deeplake.load('hub://activeloop/coco-train') # Deep Lake Dataset
ds_c.visualize()
ds_s = deeplake.load('hub://activeloop/wiki-art')
ds_s.visualize()

tform = transforms.Compose([
    transforms.ToPILImage(), # Must convert to PIL image for subsequent operations to run
    transforms.Resize((224, 224)),
    transforms.ToTensor(), # Must convert to pytorch tensor for subsequent operations to run
])

# content Dataloader
c_loader= ds_c.pytorch(batch_size = 4, transform = {'images': tform}, shuffle = False)
c_iter = iter(c_loader)

# style dataloader
s_loader= ds_s.pytorch(batch_size = 1, transform = {'images': tform, 'labels': None}, shuffle = False)
s_iter = iter(s_loader)

max_iter = 500
vgg19 = models.vgg19(pretrained=True)
loss_factor = 10
device = torch.device('cuda')  

train(model, vgg19, c_iter, s_iter, device, max_iter)
