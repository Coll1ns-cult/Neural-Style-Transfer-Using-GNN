from DPT.classification.models.dpt.box_coder import *
from DPT.classification.models.dpt.box_coder import Simple_DePatch
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


def _build_patch_embeds(embed_dims, img_size, Depatch):
    patch_embeds=[]
    for i in range(4):
        inchans = embed_dims[i-1] if i>0 else 3
        in_size = img_size // 2**(i+1) if i>0 else img_size
        patch_size = 2 if i > 0 else 4
        if Depatch[i]:
            box_coder = pointwhCoder(input_size=in_size, patch_count=in_size//patch_size, weights=(1.,1.,1.,1.), pts=3, tanh=True, wh_bias=torch.tensor(5./3.).sqrt().log())
            patch_embeds.append(
                Simple_DePatch(box_coder, img_size=in_size, patch_size=patch_size, patch_pixel=3, patch_count=in_size//patch_size,
                 in_chans=inchans, embed_dim=embed_dims[i], another_linear=True, use_GE=True, with_norm=True))
        else:
            patch_embeds.append(
                PatchEmbed(img_size=in_size, patch_size=patch_size, in_chans=inchans,
                           embed_dim=embed_dims[i]))
    return patch_embeds


