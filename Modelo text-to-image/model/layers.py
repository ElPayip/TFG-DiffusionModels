import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many
from einops_exts.torch import EinopsToAndFrom
from model.t5 import get_encoded_dim, prob_mask_like
import math



class LayerNorm(nn.Module):
    """
    LayerNorm
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x: torch.tensor) -> torch.tensor:
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)



class SinusoidalPosEmb(nn.Module):
    '''
    Generates sinusoidal positional embedding tensor. In this case, position corresponds to time. For more information
        on sinusoidal embeddings, see ["Positional Encoding - Additional Details"](https://www.assemblyai.com/blog/how-imagen-actually-works/#timestep-conditioning).
    '''
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        :param x: Tensor of positions (i.e. times) to generate embeddings for.
        :return: T x D tensor where T is the number of positions/times and D is the dimensionality of the embedding
            space
        """
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = rearrange(x, 'i -> i 1') * rearrange(emb, 'j -> 1 j')
        return torch.cat((emb.sin(), emb.cos()), dim=-1).squeeze()



class TimeConditioningLayer(nn.Module):
  def __init__(self, dim, emb_dim=None):
    super().__init__()
    NUM_TIME_TOKENS = 2  # Number of time tokens to use in conditioning tensor
    if emb_dim is None:
      emb_dim = dim

    # Maps time to time hidden state
    self.to_time_hiddens = nn.Sequential(
        SinusoidalPosEmb(dim),
        nn.Linear(dim, emb_dim),
        nn.ReLU()
    )
    # Maps time hidden state to time conditioning (non-attention)
    self.to_time_cond = nn.Sequential(
        nn.Linear(emb_dim, emb_dim)
    )
    # Maps time hidden states to time tokens for main conditioning tokens (attention)
    self.to_time_tokens = nn.Sequential(
        nn.Linear(emb_dim, dim * NUM_TIME_TOKENS),
        Rearrange('b (r d) -> b r d', r=NUM_TIME_TOKENS)
    )

  def forward(self, time: torch.tensor, ) -> tuple[torch.tensor, torch.tensor]:
    '''
    Generate t and time_tokens

    :param time: Tensor of shape (b,). The timestep for each image in the batch.
    :return: tuple(t, time_tokens)
        t: Tensor of shape (b, time_cond_dim) where `time_cond_dim` is 4x the UNet `dim`, or 8 if conditioning
        on lowres image.
        time_tokens: Tensor of shape (b, NUM_TIME_TOKENS, dim), where `NUM_TIME_TOKENS` defaults to 2.
    '''
    time_hiddens = self.to_time_hiddens(time)
    t = self.to_time_cond(time_hiddens)
    time_tokens = self.to_time_tokens(time_hiddens)
    return t, time_tokens



class TextConditioningLayer(nn.Module):
  def __init__(self, dim, emb_dim=None, max_text_len=512):
    super().__init__()
    self.max_text_len = max_text_len
    if emb_dim is None:
      emb_dim = 4*dim

    self.norm_cond = nn.LayerNorm(dim)
    self.text_embed_dim = get_encoded_dim()
    self.text_to_cond = nn.Linear(self.text_embed_dim, dim)

    self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, dim))
    self.null_text_hidden = nn.Parameter(torch.randn(1, emb_dim))

    # For injecting text information into time conditioning (non-attention)
    self.to_text_non_attn_cond = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, emb_dim),
        nn.ReLU(),
        nn.Linear(emb_dim, emb_dim)
    )

  def forward(
            self,
            text_embeds: torch.tensor,
            cond_drop_prob: float,
            device: torch.device,
            text_mask: torch.tensor,
            t: torch.tensor,
            time_tokens: torch.tensor
    ) -> tuple[torch.tensor, torch.tensor]:
        '''
        Condition on text.

        :param text_embeds: Text embedding from T5 encoder. Shape (b, mw, ed), where

            :code:`b` is the batch size,

            :code:`mw` is the maximum number of words in a caption in the batch, and

            :code:`ed` is the T5 encoding dimension.
        :param cond_drop_prob: Probability of conditional dropout for `classifier-free guidance <https://www.assemblyai.com/blog/how-imagen-actually-works/#classifier-free-guidance>`_
        :param device: Device to use.
        :param text_mask: Text mask for text embeddings. Shape (b, minimagen.t5.MAX_LENGTH)
        :param t: Time conditioning tensor.
        :param time_tokens: Time conditioning tokens.
        :return: tuple(t, c)

            :code:`t`: Time conditioning tensor

            :code:`c`: Main conditioning tokens
        '''
        text_tokens = None
        if text_embeds is not None:

            # Project the text embeddings to the conditioning dimension `cond_dim`.
            text_tokens = self.text_to_cond(text_embeds)

            # Truncate the tokens to have the maximum number of allotted words.
            text_tokens = text_tokens[:, :self.max_text_len]

            # Pad the text tokens up to self.max_text_len if needed
            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len
            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))

            # Prob. mask for clf-free guidance conditional dropout. Tells which elts in the batch to keep. Size (b,).
            text_keep_mask = prob_mask_like((text_embeds.shape[0],), 1 - cond_drop_prob, device=device)
            # Combines T5 and clf-free guidance masks
            text_keep_mask_embed = rearrange(text_keep_mask, 'b -> b 1 1')
            if text_mask is not None:
                if remainder > 0:
                    text_mask = F.pad(text_mask, (0, remainder), value=False)

                text_mask = rearrange(text_mask, 'b n -> b n 1')  # (b, self.max_text_len, 1)
                text_keep_mask_embed = text_mask & text_keep_mask_embed  # (b, self.max_text_len, 1)

            # Creates NULL tensor of size (1, self.max_text_len, cond_dim)
            null_text_embed = self.null_text_embed.to(text_tokens.dtype)  # for some reason pytorch AMP not working

            # Replaces masked elements with NULL
            text_tokens = torch.where(
                text_keep_mask_embed,
                text_tokens,
                null_text_embed
            )

            # Extra non-attention conditioning by projecting and then summing text embeddings to time (text hiddens)
            # Pool the text tokens along the word dimension.
            mean_pooled_text_tokens = text_tokens.mean(dim=-2)

            # Project to `time_cond_dim`
            text_hiddens = self.to_text_non_attn_cond(mean_pooled_text_tokens)  # (b, cond_dim) -> (b, time_cond_dim)

            null_text_hidden = self.null_text_hidden.to(t.dtype)

            # Drop relevant conditioning info as demanded by clf-free guidance mask
            text_keep_mask_hidden = rearrange(text_keep_mask, 'b -> b 1')
            text_hiddens = torch.where(
                text_keep_mask_hidden,
                text_hiddens,
                null_text_hidden
            )

            # Add this conditioning to our `t` tensor
            t = t + text_hiddens
        # main conditioning tokens `c` - concatenate time/text tokens
        c = time_tokens if text_tokens is None else torch.cat((time_tokens, text_tokens), dim=-2)
        # normalize conditioning tokens
        c = self.norm_cond(c)
        return t, c



class CrossAttention(nn.Module):
    """
    Multi-headed cross attention.
    """

    def __init__(
            self,
            dim: int,
            *,
            context_dim: int = None,
            dim_head: int = 64,
            heads: int = 8,
            norm_context: bool = False
    ):
        """
        :param dim: Input dimensionality.
        :param context_dim: Context dimensionality.
        :param dim_head: Dimensionality for each attention head.
        :param heads: Number of attention heads.
        :param norm_context: Whether to LayerNorm the context.
        """
        super().__init__()
        self.scale = dim_head ** -0.5   # 1/sqrt(dh)
        self.heads = heads
        inner_dim = dim_head * heads

        context_dim = context_dim if context_dim is not None else dim

        self.norm = LayerNorm(dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else None

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x: torch.tensor, context: torch.tensor, mask: torch.tensor = None) -> torch.tensor:
        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        if self.norm_context is not None:
          context = self.norm_context(context)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h=self.heads)

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b h 1 d', h=self.heads, b=b)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        max_neg_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
  


class SelfAttention(nn.Module):
    """
    Multi-headed self-attention
    """

    def __init__(
            self,
            dim: int,
            *,
            dim_head: int = 64,
            heads: int = 8,
            context_dim: int = None
    ):
        """
        :param dim: Input dimensionality.
        :param dim_head: Dimensionality for each attention head.
        :param heads: Number of attention heads.
        :param context_dim: Context dimensionality.
        """
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(2, dim_head))
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)

        self.to_context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if context_dim is not None else None

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            LayerNorm(dim)
        )

    def forward(self, x: torch.tensor, context: torch.tensor = None, mask: torch.tensor = None,
                attn_bias: torch.tensor = None) -> torch.tensor:

        b, n, device = *x.shape[:2], x.device

        x = self.norm(x)
        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        q = q * self.scale

        # add null key / value for classifier free guidance in prior net

        nk, nv = repeat_many(self.null_kv.unbind(dim=-2), 'd -> b 1 d', b=b)
        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        # add text conditioning, if present

        if context is not None:
            assert self.to_context is not None
            ck, cv = self.to_context(context).chunk(2, dim=-1)
            k = torch.cat((ck, k), dim=-2)
            v = torch.cat((cv, v), dim=-2)

        # calculate query / key similarities

        sim = einsum('b h i d, b j d -> b h i j', q, k)

        # relative positional encoding (T5 style)

        if attn_bias is not None:
            sim = sim + attn_bias

        # masking

        max_neg_value = -torch.finfo(sim.dtype).max

        if mask is not None:
            mask = F.pad(mask, (1, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        # attention

        attn = sim.softmax(dim=-1, dtype=torch.float32)

        # aggregate values

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)



class Block(nn.Module):
  def __init__(self, in_channels, out_channels, time_cond_dim=None, text_cond_dim=None, device='cuda', residual=True):
    super().__init__()
    self.residual = residual

    self.time_fc = nn.Sequential(
        nn.Linear(time_cond_dim, 2*out_channels),
        nn.ReLU()
    ) if time_cond_dim is not None else None

    self.attn = EinopsToAndFrom('b c h w', 'b (h w) c',
                CrossAttention(dim=out_channels, context_dim=text_cond_dim, heads=4)
    ) if text_cond_dim is not None else None

    # First convolutional layer
    self.conv1 = nn.Sequential(
        nn.BatchNorm2d(in_channels),   # Batch normalization
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, device=device),   # 3x3 kernel with stride 1 and padding 1
        nn.ReLU()   # ReLU activation function
        )

    self.norm = nn.BatchNorm2d(out_channels)
    # Second convolutional layer
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, 3, 1, 1),   # 3x3 kernel with stride 1 and padding 1
        nn.ReLU()   # ReLU activation function
        )
    
    self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels and residual else None

  def forward(self, x, t=None, c=None):
    x0 = x
    x = self.conv1(x)

    if self.attn is not None and c is not None:
        x = x + self.attn(x, context=c)   # Residual cross-attention
    
    x = self.norm(x)

    if self.time_fc is not None and t is not None:
        t = self.time_fc(t)
        t = rearrange(t, 'b c -> b c 1 1')  # (b, 2c, 1, 1)
        scale, shift = t.chunk(2, dim=1)    # (b, c, 1, 1) *2
        x = x * (scale + 1) + shift

    return self.conv2(x) + (self.res_conv(x0) if self.res_conv is not None else x0 if self.residual else 0)
  


class TransformerBlock(nn.Module):
    """
    Transformer encoder block. Responsible for applying attention at the end of a chain of :class:`.ResnetBlock`s at
        each layer in the U-Met.
    """
    def __init__(
            self,
            dim: int,
            *,
            heads: int = 8,
            dim_head: int = 32,
            ff_mult: int = 2,
            context_dim: int = None,
            do_ff: bool = True
    ):
        """

        :param dim: Number of channels in the input.
        :param heads: Number of attention heads for multi-headed :class:`.Attention`.
        :param dim_head: Dimensionality for each attention head in multi-headed :class:`.Attention`.
        :param ff_mult: Channel depth multiplier for the :class:`.ChanFeedForward` MLP applied after multi-headed
            attention.
        :param context_dim: Dimensionality of the context.
        """
        super().__init__()
        self.attn = EinopsToAndFrom('b c h w', 'b (h w) c',
                                    SelfAttention(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim))
        self.ff = nn.Sequential(  # Feed forward
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, ff_mult*dim, 1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(ff_mult*dim),
            nn.Conv2d(ff_mult*dim, dim, 1, bias=False),
            nn.ReLU()
        ) if do_ff else None

    def forward(self, x: torch.tensor, context: torch.tensor = None) -> torch.tensor:
        x = self.attn(x, context=context) + x
        if self.ff is not None:
          x = self.ff(x) + x
        return x



class UnetUp(nn.Module):
  def __init__(self, in_channels, out_channels, time_cond_dim=None, text_cond_dim=None, device=None, self_attn=False, residual=True):
    super(UnetUp, self).__init__()
    self.skip_connect_scale = 2**-0.5
    
    # The model consists of a ConvTranspose2d layer for upsampling, followed by two ResidualConvBlock layers
    self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
    self.b1 = Block(out_channels, out_channels, time_cond_dim=time_cond_dim, text_cond_dim=text_cond_dim, device=device, residual=residual)
    self.b2 = Block(out_channels, out_channels, time_cond_dim=time_cond_dim, device=device, residual=residual)
    self.attn = TransformerBlock(out_channels, heads=4, dim_head=out_channels//2) if self_attn else None

  def forward(self, x, skip, t=None, c=None):
    # Concatenate the input tensor x with the skip connection tensor along the channel dimension
    #skip = skip * self.skip_connect_scale
    x = torch.cat((x, skip), dim=1)

    # Pass the concatenated tensor through the sequential model and return the output
    x = self.up(x)
    x = self.b1(x, t=t, c=c)
    x = self.b2(x, t=t, c=c)
    if self.attn is not None:
       x = self.attn(x)
    return x



class UnetDown(nn.Module):
  def __init__(self, in_channels, out_channels, time_cond_dim=None, text_cond_dim=None, device=None, self_attn=False, residual=True):
    super(UnetDown, self).__init__()
    # Each block consists of two Block layers, followed by a MaxPool2d layer for downsampling
    self.b1 = Block(in_channels, out_channels, time_cond_dim=time_cond_dim, text_cond_dim=text_cond_dim, device=device, residual=residual)
    self.b2 = Block(out_channels, out_channels, time_cond_dim=time_cond_dim, device=device, residual=residual)
    self.attn = TransformerBlock(out_channels, heads=4, dim_head=out_channels//2) if self_attn else None
    self.maxpool = nn.MaxPool2d(2)

  def forward(self, x, t=None, c=None):
    x = self.b1(x, t, c)
    x = self.b2(x, t, c)
    if self.attn is not None:
       x = self.attn(x)
    return self.maxpool(x)