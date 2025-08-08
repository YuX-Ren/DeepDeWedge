import torch
import torch.nn as nn
from einops import rearrange, repeat
import math

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


def compute_mhsa(q, k, v, scale_factor=1, mask=None):
    # resulted shape will be: [batch, heads, tokens, tokens]
    scaled_dot_prod = torch.einsum('... i d , ... j d -> ... i j', q, k) / math.sqrt(q.shape[-1])

    if mask is not None:
        mask = (torch.bmm(mask.unsqueeze(-1),mask.unsqueeze(1))==0.).unsqueeze(1)
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -1e16)

    attention = torch.softmax(scaled_dot_prod, dim=-1)
    # calc result per head
    return torch.einsum('... i j , ... j d -> ... i d', attention, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        """
        Implementation of multi-head attention layer of the original transformer model.
        einsum and einops.rearrange is used whenever possible
        Args:
            dim: token's dimension, i.e. word embedding vector size
            heads: the number of distinct representations to learn
            dim_head: the dim of the head. In general dim_head<dim.
            However, it may not necessary be (dim/heads)
        """
        super().__init__()
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        _dim = self.dim_head * heads
        self.heads = heads
        self.to_qvk = nn.Linear(dim, _dim * 3, bias=False)
        self.W_0 = nn.Linear(_dim, dim, bias=False)
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x, mask=None):
        assert x.dim() == 3
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3*heads ]

        # decomposition to q,v,k and cast to tuple
        # the resulted shape before casting to tuple will be: [3, batch, heads, tokens, dim_head]
        q, k, v = tuple(rearrange(qkv, 'b t (d k h ) -> k b h t d ', k=3, h=self.heads))

        out = compute_mhsa(q, k, v, mask=mask, scale_factor=self.scale_factor)

        # re-compose: merge heads with dim_head
        out = rearrange(out, "b h t d -> b t (h d)")
        # Apply final linear transformation layer
        return self.W_0(out)
    


class Embeddings3D(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size=16, dropout=0.1):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
                                          kernel_size=patch_size, stride=patch_size, bias=False)
        self.position_embeddings = AbsPositionalEncoding1D(self.n_patches, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x is a 5D tensor
        """
        x = rearrange(self.patch_embeddings(x), 'b d x y z -> b (x y z) d')
        embeddings = self.dropout(self.position_embeddings(x))
        return embeddings



class AbsPositionalEncoding1D(nn.Module):
    def __init__(self, tokens, dim):
        super(AbsPositionalEncoding1D, self).__init__()
        self.abs_pos_enc = nn.Parameter(torch.randn(1,tokens, dim))

    def forward(self, x):
        batch = x.size()[0]
        return x + expand_to_batch(self.abs_pos_enc, desired_size=batch)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout, extract_layers, dim_linear_block):
        super().__init__()
        self.layer = nn.ModuleList()  # Unused, kept for compatibility
        self.extract_layers = extract_layers
        self.block_list = nn.ModuleList()
        for _ in range(num_layers):
            self.block_list.append(
                TransformerBlock(dim=embed_dim, heads=num_heads, dim_linear_block=dim_linear_block, dropout=dropout,
                                 prenorm=False))

    def forward(self, x):
        for layer_block in self.block_list:
            x = layer_block(x)  # Removed seq and mask
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None, dim_linear_block=1024, dropout=0.1, activation=nn.GELU,
                 prenorm=False):
        super().__init__()
        self.mhsa = MultiHeadSelfAttention(dim=dim, heads=heads, dim_head=dim_head)
        self.prenorm = prenorm
        self.drop = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim)
        self.norm_2 = nn.LayerNorm(dim)

        self.linear = nn.Sequential(
            nn.Linear(dim, dim_linear_block),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_linear_block, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        if self.prenorm:
            x = self.drop(self.mhsa(self.norm_1(x))) + x
            x = self.linear(self.norm_2(x)) + x
        else:
            x = self.norm_1(self.drop(self.mhsa(x)) + x)
            x = self.norm_2(self.linear(x) + x)
        return x

class Vit3D(nn.Module):
    def __init__(self, img_shape=(360, 360, 360), input_dim=1, output_dim=4, embed_dim=768, patch_size=36,
                 num_heads=6, dropout=0.1, ext_layers=[3, 6, 9, 12], norm="instance",
                 dim_linear_block=3072, decoder_dim=256):
        super().__init__()
        self.num_layers = 8
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.ext_layers = ext_layers
        self.decoder_dim = decoder_dim

        self.norm = nn.BatchNorm3d if norm == 'batch' else nn.InstanceNorm3d
        self.embed = Embeddings3D(input_dim=input_dim, embed_dim=embed_dim, cube_size=img_shape,
                                  patch_size=patch_size, dropout=dropout)
        
        self.transformer = TransformerEncoder(embed_dim, num_heads, self.num_layers, dropout, ext_layers,
                                             dim_linear_block=dim_linear_block)
        self.out = nn.Linear(embed_dim, 12)
        self.to_hV = nn.Linear(embed_dim, decoder_dim)
        self.atom_norm = nn.LayerNorm(12)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)  # Replaced cross_attn

    def forward(self, x):
        batch_size = x.shape[0]
        n_patches = int((self.img_shape[0] * self.img_shape[1] * self.img_shape[2]) / (self.patch_size ** 3))

        transformer_input = self.embed(x)
        protein = self.transformer(transformer_input)
        y = self.self_attn(protein)  # Replaced cross-attention with self-attention
        h_V = self.to_hV(y)

        return h_V


if __name__ == "__main__":
    model = Vit3D(img_shape=(96, 96, 96),patch_size=4).to("cuda:0")
    x = torch.randn(1, 1, 96, 96, 96).to("cuda:0")
    output = model(x)
    print(output.shape)