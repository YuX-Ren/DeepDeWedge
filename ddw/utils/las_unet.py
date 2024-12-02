import torch
import torch.nn as nn
from .las_utils import *

class EDM_Unet3D(torch.nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 64,
        num_downsample_layers: int = 3, 
        drop_prob: float = 0.0,
        sigma_min       = 0,                
        sigma_max       = float('inf'),     
        sigma_data      = 1.0,              
        normalization_loc = 0.0,
        normalization_scale = 1.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_downsample_layers = num_downsample_layers
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.normalization_loc = normalization_loc
        self.normalization_scale = normalization_scale
        self.__init_model__()

    @property
    def normalization_loc(self):
        return self._normalization_loc

    @normalization_loc.setter
    def normalization_loc(self, normalization_loc):
        self._normalization_loc = nn.parameter.Parameter(
            torch.tensor(normalization_loc), requires_grad=False
        )

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @normalization_scale.setter
    def normalization_scale(self, normalization_scale):
        self._normalization_scale = nn.parameter.Parameter(
            torch.tensor(normalization_scale), requires_grad=False
        )
        self.sigma_data = self._normalization_scale.item()


    def __init_model__(self):
        self.model = UNetModel(
            in_channels=self.in_chans,
            out_channels=self.out_chans,
            base_channels=self.chans,
        )


    def normalize(self, volume: torch.Tensor) -> torch.Tensor:
        return (volume - self.normalization_loc) / (self.normalization_scale + 1e-6)

    def denormalize(self, volume: torch.Tensor) -> torch.Tensor:
        return volume * (self.normalization_scale + 1e-6) + self.normalization_loc
    
    def forward(self, x, sigma,  **model_kwargs):
        x = x
        sigma = sigma.reshape(-1, 1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        F_x = self.model((c_in * x), c_noise.flatten(), **model_kwargs)
        D_x = c_skip * x + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class Unet3D(torch.nn.Module):
    """
    PyTorch implementation of a 3D U-Net, which was inspired by the one used in the IsoNet software package (https://github.com/IsoNet-cryoET/IsoNet/tree/master/models/unet)
    """

    def __init__(
        self,
        in_chans: int = 1,
        out_chans: int = 1,
        chans: int = 32,
        num_downsample_layers: int = 3,
        drop_prob: float = 0.0,
        residual: bool = True,
        normalization_loc: float = 0.0,
        normalization_scale: float = 1.0,
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_downsample_layers = num_downsample_layers
        self.drop_prob = drop_prob
        self.residual = residual
        self.normalization_loc = normalization_loc
        self.normalization_scale = normalization_scale
        self.__init_model__()

    @property
    def normalization_loc(self):
        return self._normalization_loc

    @normalization_loc.setter
    def normalization_loc(self, normalization_loc):
        self._normalization_loc = nn.parameter.Parameter(
            torch.tensor(normalization_loc), requires_grad=False
        )

    @property
    def normalization_scale(self):
        return self._normalization_scale

    @normalization_scale.setter
    def normalization_scale(self, normalization_scale):
        self._normalization_scale = nn.parameter.Parameter(
            torch.tensor(normalization_scale), requires_grad=False
        )

    def __init_model__(self):
        self.model = UNetModel(
            in_channels=self.in_chans,
            out_channels=self.out_chans,
            base_channels=self.chans,
        )


    def normalize(self, volume: torch.Tensor) -> torch.Tensor:
        return (volume - self.normalization_loc) / (self.normalization_scale + 1e-6)

    def denormalize(self, volume: torch.Tensor) -> torch.Tensor:
        return volume * (self.normalization_scale + 1e-6) + self.normalization_loc

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        volume = self.normalize(volume)

        output = volume

        output = self.model(output)
        if self.residual:
            output = output + volume

        output = self.denormalize(output)
        return output

class UNetModel(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 image_size: int = 64,
                 base_channels: int = 32,
                 dim_mults=(1, 2, 4, 8),
                 dropout: float = 0.0,
                 num_heads: int = 1,
                 world_dims: int = 3,
                 attention_resolutions=(4, 8),
                 with_attention: bool = False,
                 verbose: bool = False,
                 image_condition_dim: int = VIT_FEATURE_CHANNEL,
                 text_condition_dim: int = CLIP_FEATURE_CHANNEL,
                 kernel_size: float = 1.0,
                 use_sketch_condition: bool = False,
                 use_text_condition: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 ):
        super().__init__()
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        channels = [base_channels, *
                    map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose
        emb_dim = base_channels * 4

        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        )
        if self.use_text_condition:
            self.text_emb = nn.Sequential(
                nn.Linear(text_condition_dim, emb_dim),
                activation_function(),
                nn.Linear(emb_dim, emb_dim)
            )

        self.input_emb = conv_nd(world_dims, 1, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            self.downs.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_in, dim_out,
                            emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition),
                CrossAttention(feature_dim=dim_out, sketch_dim=image_condition_dim,
                               kernel_size=kernel_size, vit_local=vit_local, vit_global=vit_global,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross and self.use_sketch_condition else our_Identity(),
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Downsample(
                    dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition)
        
        self.mid_cross_attn = CrossAttention(feature_dim=mid_dim, sketch_dim=image_condition_dim, vit_local=vit_local, vit_global=vit_global,
                                             kernel_size=kernel_size,
                                             num_heads=num_heads, image_size=res, world_dims=world_dims,
                                             drop_out=dropout) if self.use_sketch_condition else our_Identity()
        self.mid_self_attn = nn.Sequential(
            normalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions and with_attention else our_Identity()
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            self.ups.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_out * 2, dim_in,
                            emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition),
                CrossAttention(feature_dim=dim_in, sketch_dim=image_condition_dim,
                               kernel_size=kernel_size, vit_local=vit_local, vit_global=vit_global,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross and self.use_sketch_condition else our_Identity(),
                nn.Sequential(
                    normalization(dim_in),
                    activation_function(),
                    AttentionBlock(
                        dim_in, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Upsample(
                    dim_in, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds //= 2

        self.end = nn.Sequential(
            normalization(base_channels),
            activation_function()
        )

        self.out = conv_nd(world_dims, base_channels, 1, 3, padding=1)

    def forward(self, x, t=None, img_condition=None, text_condition=None, projection_matrix=None, x_self_cond=None, kernel_size=None):
        # normalize input
        if self.verbose:
            print("input size:")
            print(x.shape)

        x = self.input_emb(x)
        if t is not None:
            t = self.time_emb(self.time_pos_emb(t))
            if self.use_text_condition:
                text_condition = self.text_emb(text_condition)
            h = []

            for resnet, cross_attn, self_attn, downsample in self.downs:
                x = resnet(x, t, text_condition)
                if self.verbose:
                    print(x.shape)
                    if type(cross_attn) == CrossAttention:
                        print("cross attention at resolution: ",
                            cross_attn.image_size)
                x = cross_attn(x, img_condition,  projection_matrix, kernel_size)
                x = self_attn(x)
                if self.verbose:
                    print(x.shape)
                h.append(x)
                x = downsample(x)
                if self.verbose:
                    print(x.shape)

            if self.verbose:
                print("enter bottle neck")
            x = self.mid_block1(x, t, text_condition)
            if self.verbose:
                print(x.shape)

            x = self.mid_cross_attn(
                x, img_condition, projection_matrix, kernel_size)
            x = self.mid_self_attn(x)
            if self.verbose:
                print("cross attention at resolution: ",
                    self.mid_cross_attn.image_size)
                print(x.shape)
            x = self.mid_block2(x, t, text_condition)
            if self.verbose:
                print(x.shape)
                print("finish bottle neck")

            for resnet, cross_attn, self_attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                if self.verbose:
                    print(x.shape)
                x = resnet(x, t, text_condition)
                if self.verbose:
                    print(x.shape)
                x = cross_attn(x, img_condition, projection_matrix, kernel_size)
                x = self_attn(x)
                if self.verbose:
                    if type(cross_attn) == CrossAttention:
                        print("cross attention at resolution: ",
                            cross_attn.image_size)
                    print(x.shape)
                x = upsample(x)
                if self.verbose:
                    print(x.shape)
        else:
            t = None
            if self.use_text_condition:
                text_condition = self.text_emb(text_condition)
            h = []

            for resnet, cross_attn, self_attn, downsample in self.downs:
                x = resnet(x, t, text_condition)
                if self.verbose:
                    print(x.shape)
                    if type(cross_attn) == CrossAttention:
                        print("cross attention at resolution: ",
                            cross_attn.image_size)
                x = cross_attn(x, img_condition,  projection_matrix, kernel_size)
                x = self_attn(x)
                if self.verbose:
                    print(x.shape)
                h.append(x)
                x = downsample(x)
                if self.verbose:
                    print(x.shape)

            if self.verbose:
                print("enter bottle neck")
            x = self.mid_block1(x, t, text_condition)
            if self.verbose:
                print(x.shape)

            x = self.mid_cross_attn(
                x, img_condition, projection_matrix, kernel_size)
            x = self.mid_self_attn(x)
            if self.verbose:
                print("cross attention at resolution: ",
                    self.mid_cross_attn.image_size)
                print(x.shape)
            x = self.mid_block2(x, t, text_condition)
            if self.verbose:
                print(x.shape)
                print("finish bottle neck")

            for resnet, cross_attn, self_attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                if self.verbose:
                    print(x.shape)
                x = resnet(x, t, text_condition)
                if self.verbose:
                    print(x.shape)
                x = cross_attn(x, img_condition, projection_matrix, kernel_size)
                x = self_attn(x)
                if self.verbose:
                    if type(cross_attn) == CrossAttention:
                        print("cross attention at resolution: ",
                            cross_attn.image_size)
                    print(x.shape)
                x = upsample(x)
                if self.verbose:
                    print(x.shape)
        x = self.end(x)
        if self.verbose:
            print(x.shape)

        return self.out(x)


class UNetModelEncoder(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 out_channels: int = 1,
                 image_size: int = 64,
                 base_channels: int = 64,
                 dim_mults=(1, 2, 4, 8),
                 dropout: float = 0.1,
                 num_heads: int = 1,
                 world_dims: int = 3,
                 attention_resolutions=(4, 8),
                 with_attention: bool = True,
                 verbose: bool = False,
                 image_condition_dim: int = VIT_FEATURE_CHANNEL,
                 text_condition_dim: int = CLIP_FEATURE_CHANNEL,
                 kernel_size: float = 1.0,
                 use_sketch_condition: bool = False,
                 use_text_condition: bool = False,
                 vit_global: bool = False,
                 vit_local: bool = True,
                 ):
        super().__init__()
        self.use_sketch_condition = use_sketch_condition
        self.use_text_condition = use_text_condition
        channels = [base_channels, *
                    map(lambda m: base_channels * m, dim_mults)]
        in_out = list(zip(channels[:-1], channels[1:]))

        self.verbose = verbose
        emb_dim = base_channels * 4

        self.time_pos_emb = LearnedSinusoidalPosEmb(base_channels)
        self.time_emb = nn.Sequential(
            nn.Linear(base_channels + 1, emb_dim),
            activation_function(),
            nn.Linear(emb_dim, emb_dim)
        )
        if self.use_text_condition:
            self.text_emb = nn.Sequential(
                nn.Linear(text_condition_dim, emb_dim),
                activation_function(),
                nn.Linear(emb_dim, emb_dim)
            )

        self.input_emb = conv_nd(world_dims, 2, base_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        ds = 1

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            res = image_size // ds
            use_cross = (res == 4 or res == 8)
            self.downs.append(nn.ModuleList([
                ResnetBlock(world_dims, dim_in, dim_out,
                            emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition),
                CrossAttention(feature_dim=dim_out, sketch_dim=image_condition_dim,
                               kernel_size=kernel_size, vit_local=vit_local, vit_global=vit_global,
                               num_heads=num_heads, image_size=res, world_dims=3,
                               drop_out=dropout) if use_cross and self.use_sketch_condition else our_Identity(),
                nn.Sequential(
                    normalization(dim_out),
                    activation_function(),
                    AttentionBlock(
                        dim_out, num_heads=num_heads)) if ds in attention_resolutions and with_attention else our_Identity(),
                Downsample(
                    dim_out, dims=world_dims) if not is_last else our_Identity()
            ]))
            if not is_last:
                ds *= 2

        mid_dim = channels[-1]
        res = image_size // ds
        self.mid_block1 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition)
        
        self.mid_cross_attn = CrossAttention(feature_dim=mid_dim, sketch_dim=image_condition_dim, vit_local=vit_local, vit_global=vit_global,
                                             kernel_size=kernel_size,
                                             num_heads=num_heads, image_size=res, world_dims=world_dims,
                                             drop_out=dropout) if self.use_sketch_condition else our_Identity()
        self.mid_self_attn = nn.Sequential(
            normalization(mid_dim),
            activation_function(),
            AttentionBlock(mid_dim, num_heads=num_heads)
        ) if ds in attention_resolutions and with_attention else our_Identity()
        self.mid_block2 = ResnetBlock(
            world_dims, mid_dim, mid_dim, emb_dim=emb_dim, dropout=dropout, use_text_condition=use_text_condition)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.out = nn.Sequential(
            nn.Linear(mid_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x, t=None, img_condition=None, text_condition=None, projection_matrix=None, x_self_cond=None, kernel_size=None):
        # normalize input
        bs, c, *spatial = x.shape

        if self.verbose:
            print("input size:")
            print(x.shape)

        x = self.input_emb(x)
        if t is not None:
            t = self.time_emb(self.time_pos_emb(t))
            if self.use_text_condition:
                text_condition = self.text_emb(text_condition)
            h = []

            for resnet, cross_attn, self_attn, downsample in self.downs:
                x = resnet(x, t, text_condition)
                if self.verbose:
                    print(x.shape)
                    if type(cross_attn) == CrossAttention:
                        print("cross attention at resolution: ",
                            cross_attn.image_size)
                x = cross_attn(x, img_condition,  projection_matrix, kernel_size)
                x = self_attn(x)
                if self.verbose:
                    print(x.shape)
                h.append(x)
                x = downsample(x)
                if self.verbose:
                    print(x.shape)

            if self.verbose:
                print("enter bottle neck")
            x = self.mid_block1(x, t, text_condition)
            if self.verbose:
                print(x.shape)

            x = self.mid_cross_attn(
                x, img_condition, projection_matrix, kernel_size)
            x = self.mid_self_attn(x)
            if self.verbose:
                print("cross attention at resolution: ",
                    self.mid_cross_attn.image_size)
                print(x.shape)
            x = self.mid_block2(x, t, text_condition)
            if self.verbose:
                print(x.shape)
                print("finish bottle neck")

        else:
            t = None
            if self.use_text_condition:
                text_condition = self.text_emb(text_condition)
            h = []

            for resnet, cross_attn, self_attn, downsample in self.downs:
                x = resnet(x, t, text_condition)
                if self.verbose:
                    print(x.shape)
                    if type(cross_attn) == CrossAttention:
                        print("cross attention at resolution: ",
                            cross_attn.image_size)
                x = cross_attn(x, img_condition,  projection_matrix, kernel_size)
                x = self_attn(x)
                if self.verbose:
                    print(x.shape)
                h.append(x)
                x = downsample(x)
                if self.verbose:
                    print(x.shape)

            if self.verbose:
                print("enter bottle neck")
            x = self.mid_block1(x, t, text_condition)
            if self.verbose:
                print(x.shape)

            x = self.mid_cross_attn(
                x, img_condition, projection_matrix, kernel_size)
            x = self.mid_self_attn(x)
            if self.verbose:
                print(x.shape)
            x = self.mid_block2(x, t, text_condition)
            if self.verbose:
                print(x.shape)
                print("finish bottle neck")
        x = self.pool(x).reshape(bs, -1)
        x = self.out(x)

        return x
