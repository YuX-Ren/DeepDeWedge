import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import numbers
import pandas as pd
from einops import rearrange
import time


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN for 3D.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale



##########################################################################
## Layer Norm for 3D
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight
    
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        # Use GroupNorm for 3D which is more stable
        if LayerNorm_type == 'BiasFree':
            self.body = nn.GroupNorm(1, dim, affine=True)
            # Remove bias manually for BiasFree version
            nn.init.zeros_(self.body.bias)
            self.body.bias.requires_grad = False
        else:
            self.body = nn.GroupNorm(1, dim, affine=True)
        # self.body = BiasFree_LayerNorm(dim)
    def forward(self, x):
        # return x
        # x = rearrange(x, 'b c d h w -> b d h w c')
        # return rearrange(self.body(x), 'b d h w c -> b c d h w')
        return self.body(x)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN) for 3D
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv3d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv3d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv3d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA) for 3D
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv3d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b, c, d, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) d h w -> b head c (d h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (d h w) -> b (head c) d h w', head=self.num_heads, d=d, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped volume patch embedding with 3x3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=6, stride=4, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

class final_upsample(nn.Module):
    def __init__(self, n_feat):
        super(final_upsample, self).__init__()
        # conv first and then upsample

        self.body = nn.ConvTranspose3d(n_feat, n_feat//2, kernel_size=4, stride=4, padding=0, bias=False)

    def forward(self, x):
        return self.body(x)


##########################################################################
## Resizing modules for 3D
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        # Simple strided convolution for downsampling
        self.body = nn.Conv3d(n_feat, n_feat*2, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        # Transposed convolution for upsampling
        self.body = nn.ConvTranspose3d(n_feat, n_feat//2, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer 3D -----------------------
class Restormer3D(nn.Module):
    def __init__(self, 
        inp_channels=1,  # Typically 1 for 3D medical images
        out_channels=1, 
        dim = 32,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(Restormer3D, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv3d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])


        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv3d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        

        self.up1_0 = final_upsample(int(dim*2**1))
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv3d(dim, int(dim), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv3d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 

        inp_enc_level4 = self.down3_4(out_enc_level3)        
        latent = self.latent(inp_enc_level4) 
                        
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3) 
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        # out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.up1_0(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

    # def foward_with_less_layers(self, inp_img):
    #     inp_enc_level1 = self.patch_embed(inp_img)
    #     out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
    #     inp_enc_level2 = self.down1_2(out_enc_level1)
    #     out_enc_level2 = self.encoder_level2(inp_enc_level2)

    #     inp_enc_level3 = self.down2_3(out_enc_level2)
    #     latent = self.latent(inp_enc_level3) 
                        
    #     inp_dec_level2 = self.up3_2(latent)
    #     inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
    #     inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
    #     out_dec_level2 = self.decoder_level2(inp_dec_level2) 

    #     inp_dec_level1 = self.up2_1(out_dec_level2)
    #     inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
    #     out_dec_level1 = self.decoder_level1(inp_dec_level1)

    #     out_dec_level1 = self.refinement(out_dec_level1)

    #     out_dec_level1 = self.up1_0(out_dec_level1)

    #     out_dec_level1 = self.output(out_dec_level1) + inp_img


    #     return out_dec_level1        

# Example usage:
def profile_blocks(model, vol=(1,1,64,64,64), dtype=torch.bfloat16):
    device = 'cuda'; torch.cuda.reset_peak_memory_stats()
    model, x = model.to(device, dtype), torch.randn(*vol, device=device, dtype=dtype)

    # parameter memory ─ grouped by *top‑level* child name (enc1, down1, …)
    p_mem = {}
    for n, p in model.named_parameters():
        top = n.split('.')[0]
        p_mem[top] = p_mem.get(top, 0) + p.numel()*p.element_size()

    # activation memory via forward hooks
    act_mem = {}
    def hook(name):
        def fn(_, __, out):
            torch.cuda.synchronize()
            act_mem[name] = out.numel()*out.element_size()
        return fn

    for n,m in model.named_children():   # hook only top‑level blocks
        m.register_forward_hook(hook(n))

    with torch.autocast(device_type='cuda', dtype=dtype):
        _ = model(x)

    df = (pd.DataFrame([
            dict(Block=k,
                 Param_MB = p_mem.get(k,0)/2**20,
                 Activ_MB = act_mem.get(k,0)/2**20,
                 Total_MB = (p_mem.get(k,0)+act_mem.get(k,0))/2**20)
            for k in sorted(set(list(p_mem)+list(act_mem)))
        ])
        .sort_values('Total_MB', ascending=False)
        .reset_index(drop=True))
    return df

def profile_layers_detailed(model, input_shape=(1, 1, 160, 160, 160), dtype=torch.bfloat16, num_runs=10):
    """Profile each layer's forward pass time individually"""
    device = 'cuda'
    model = model.to(device, dtype)
    x = torch.randn(*input_shape, device=device, dtype=dtype)
    
    # Warm up
    with torch.autocast(device_type='cuda', dtype=dtype):
        for _ in range(3):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    # Profile each named module
    layer_times = {}
    layer_memory = {}
    
    def create_hook(name):
        def hook_fn(module, input, output):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Memory usage
            if isinstance(output, torch.Tensor):
                mem_usage = output.numel() * output.element_size()
            elif isinstance(output, (list, tuple)):
                mem_usage = sum(o.numel() * o.element_size() for o in output if isinstance(o, torch.Tensor))
            else:
                mem_usage = 0
                
            layer_memory[name] = mem_usage / (1024**2)  # MB
            
        return hook_fn
    
    # Register hooks for all named modules
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            hook = module.register_forward_hook(create_hook(name))
            hooks.append(hook)
    
    # Time individual forward passes for specific layers
    major_components = {
        'patch_embed': model.patch_embed,
        'encoder_level1': model.encoder_level1,
        'down1_2': model.down1_2,
        'encoder_level2': model.encoder_level2,
        'down2_3': model.down2_3,
        'encoder_level3': model.encoder_level3,
        'down3_4': model.down3_4,
        'latent': model.latent,
        'up4_3': model.up4_3,
        'decoder_level3': model.decoder_level3,
        'up3_2': model.up3_2,
        'decoder_level2': model.decoder_level2,
        'up2_1': model.up2_1,
        'decoder_level1': model.decoder_level1,
        'refinement': model.refinement,
        'up1_0': model.up1_0,
        'output': model.output
    }
    
    # Measure timing for major components
    for comp_name, component in major_components.items():
        times = []
        with torch.autocast(device_type='cuda', dtype=dtype):
            # Get appropriate input for this component
            if comp_name == 'patch_embed':
                comp_input = x
            elif comp_name == 'encoder_level1':
                comp_input = model.patch_embed(x)
            elif comp_name == 'down1_2':
                comp_input = model.encoder_level1(model.patch_embed(x))
            else:
                # For other components, use full forward pass up to that point
                # This is approximate but gives us relative timing
                comp_input = torch.randn(1, 64, 40, 40, 40, device=device, dtype=dtype)
            
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                try:
                    with torch.no_grad():
                        _ = component(comp_input)
                    torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                except:
                    # Skip if component can't be run independently
                    times.append(0)
                    break
        
        layer_times[comp_name] = {
            'mean_time_ms': sum(times) / len(times) if times else 0,
            'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 if len(times) > 1 else 0
        }
    
    # Clean up hooks
    for hook in hooks:
        hook.remove()
    
    # Create detailed DataFrame
    df_detailed = pd.DataFrame([
        {
            'Layer': name,
            'Mean_Time_ms': info['mean_time_ms'],
            'Std_Time_ms': info['std_time_ms'],
            'Memory_MB': layer_memory.get(name, 0)
        }
        for name, info in layer_times.items()
    ]).sort_values('Mean_Time_ms', ascending=False)
    
    return df_detailed

def profile_forward_pass(model, input_shape=(1, 1, 160, 160, 160), dtype=torch.bfloat16, num_runs=5):
    """Profile timing during actual forward pass"""
    device = 'cuda'
    model = model.to(device, dtype)
    x = torch.randn(*input_shape, device=device, dtype=dtype)
    
    # Warm up
    with torch.autocast(device_type='cuda', dtype=dtype):
        for _ in range(3):
            _ = model(x)
    
    torch.cuda.synchronize()
    
    layer_times = {}
    
    for run in range(num_runs):
        times = {}
        
        with torch.autocast(device_type='cuda', dtype=dtype):
            with torch.no_grad():
                # Patch embedding
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_enc_level1 = model.patch_embed(x)
                torch.cuda.synchronize()
                times['patch_embed'] = (time.perf_counter() - start) * 1000
                
                # Encoder Level 1
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_enc_level1 = model.encoder_level1(inp_enc_level1)
                torch.cuda.synchronize()
                times['encoder_level1'] = (time.perf_counter() - start) * 1000
                
                # Down 1->2
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_enc_level2 = model.down1_2(out_enc_level1)
                torch.cuda.synchronize()
                times['down1_2'] = (time.perf_counter() - start) * 1000
                
                # Encoder Level 2
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_enc_level2 = model.encoder_level2(inp_enc_level2)
                torch.cuda.synchronize()
                times['encoder_level2'] = (time.perf_counter() - start) * 1000
                
                # Down 2->3
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_enc_level3 = model.down2_3(out_enc_level2)
                torch.cuda.synchronize()
                times['down2_3'] = (time.perf_counter() - start) * 1000
                
                # Encoder Level 3
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_enc_level3 = model.encoder_level3(inp_enc_level3)
                torch.cuda.synchronize()
                times['encoder_level3'] = (time.perf_counter() - start) * 1000
                
                # Down 3->4
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_enc_level4 = model.down3_4(out_enc_level3)
                torch.cuda.synchronize()
                times['down3_4'] = (time.perf_counter() - start) * 1000
                
                # Latent
                torch.cuda.synchronize()
                start = time.perf_counter()
                latent = model.latent(inp_enc_level4)
                torch.cuda.synchronize()
                times['latent'] = (time.perf_counter() - start) * 1000
                
                # Up 4->3
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_dec_level3 = model.up4_3(latent)
                torch.cuda.synchronize()
                times['up4_3'] = (time.perf_counter() - start) * 1000
                
                # Concat and reduce channels level 3
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
                inp_dec_level3 = model.reduce_chan_level3(inp_dec_level3)
                torch.cuda.synchronize()
                times['reduce_chan_level3'] = (time.perf_counter() - start) * 1000
                
                # Decoder Level 3
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_dec_level3 = model.decoder_level3(inp_dec_level3)
                torch.cuda.synchronize()
                times['decoder_level3'] = (time.perf_counter() - start) * 1000
                
                # Up 3->2
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_dec_level2 = model.up3_2(out_dec_level3)
                torch.cuda.synchronize()
                times['up3_2'] = (time.perf_counter() - start) * 1000
                
                # Concat and reduce channels level 2
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
                inp_dec_level2 = model.reduce_chan_level2(inp_dec_level2)
                torch.cuda.synchronize()
                times['reduce_chan_level2'] = (time.perf_counter() - start) * 1000
                
                # Decoder Level 2
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_dec_level2 = model.decoder_level2(inp_dec_level2)
                torch.cuda.synchronize()
                times['decoder_level2'] = (time.perf_counter() - start) * 1000
                
                # Up 2->1
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_dec_level1 = model.up2_1(out_dec_level2)
                torch.cuda.synchronize()
                times['up2_1'] = (time.perf_counter() - start) * 1000
                
                # Concat level 1
                torch.cuda.synchronize()
                start = time.perf_counter()
                inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
                torch.cuda.synchronize()
                times['concat_level1'] = (time.perf_counter() - start) * 1000
                
                # Decoder Level 1
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_dec_level1 = model.decoder_level1(inp_dec_level1)
                torch.cuda.synchronize()
                times['decoder_level1'] = (time.perf_counter() - start) * 1000
                
                # Refinement
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_dec_level1 = model.refinement(out_dec_level1)
                torch.cuda.synchronize()
                times['refinement'] = (time.perf_counter() - start) * 1000
                
                # Final upsample
                torch.cuda.synchronize()
                start = time.perf_counter()
                out_dec_level1 = model.up1_0(out_dec_level1)
                torch.cuda.synchronize()
                times['up1_0'] = (time.perf_counter() - start) * 1000
                
                # Output conv and residual
                torch.cuda.synchronize()
                start = time.perf_counter()
                output = model.output(out_dec_level1) + x
                torch.cuda.synchronize()
                times['output'] = (time.perf_counter() - start) * 1000
        
        # Accumulate times
        for layer, time_ms in times.items():
            if layer not in layer_times:
                layer_times[layer] = []
            layer_times[layer].append(time_ms)
    
    # Calculate statistics
    df_forward = pd.DataFrame([
        {
            'Layer': layer,
            'Mean_Time_ms': sum(times) / len(times),
            'Std_Time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 if len(times) > 1 else 0,
            'Min_Time_ms': min(times),
            'Max_Time_ms': max(times),
            'Percentage': (sum(times) / len(times)) / sum(sum(layer_times[l]) / len(layer_times[l]) for l in layer_times) * 100
        }
        for layer, times in layer_times.items()
    ]).sort_values('Mean_Time_ms', ascending=False)
    
    return df_forward

if __name__ == "__main__":
    print("Creating Restormer3D model...")
    model = Restormer3D(dim=64).cuda()
    
    print("\n=== Forward Pass Layer Profiling ===")
    df_forward = profile_forward_pass(model)
    print(df_forward.to_string(index=False))
    
    print("\n=== Block-wise Memory Profiling ===")
    df_blocks = profile_blocks(model)
    print(df_blocks.to_string(index=False))
    
    print("\n=== Full Model Profiling ===")
    x = torch.randn(1, 1, 160, 160, 160).cuda()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        with torch.profiler.profile(use_cuda=True, record_shapes=True) as prof:
            _ = model(x)

    # Process and display profiling output
    prof_df = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    prof_df_str = str(prof_df)

    # Extract to DataFrame for cleaner display
    parsed = [line.split() for line in prof_df_str.strip().split("\n")[6:-1]]
    headers = ["Name", "Self CPU total", "CPU total", "CPU time avg", "CUDA total", "CUDA time avg", "Calls"]
    rows = []
    for row in parsed:
        if len(row) >= 7:
            rows.append({
                "Name": row[0],
                "Self CPU total": row[1],
                "CPU total": row[2],
                "CPU time avg": row[3],
                "CUDA total": row[4],
                "CUDA time avg": row[5],
                "Calls": row[6],
            })

    df_profiler = pd.DataFrame(rows)
    # sort by self_cpu_time_total
    df_profiler = df_profiler.sort_values(by="CUDA time avg", ascending=False)
    print(df_profiler.to_string(index=False))