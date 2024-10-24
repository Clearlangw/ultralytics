import torch
from torch import nn
import torch.nn.functional as F
import math
from torch.nn import init


class CrossAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h,sr_ratio, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model  =  channel
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(CrossAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.num_heads = h

        self.scale = self.d_v ** -0.5

        # key, query, value projections for all heads
        self.q = nn.Linear(d_model, d_model)
        
        self.kv_rgb = nn.Linear(d_model, d_model * 2)
        self.kv_ir = nn.Linear(d_model, d_model * 2)

        self.attn_drop_rgb = nn.Dropout(attn_pdrop)
        self.attn_drop_ir = nn.Dropout(attn_pdrop)


        self.proj_rgb = nn.Linear(d_model, d_model)
        self.proj_ir = nn.Linear(d_model, d_model)



        self.proj_drop_rgb = nn.Dropout(resid_pdrop)
        self.proj_drop_ir = nn.Dropout(resid_pdrop)

        # self.kv = nn.Linear(d_model, d_model * 2)
        self.out_rgb = nn.Conv2d(d_model*2, d_model, kernel_size=1, stride=1)
        self.out_ir = nn.Conv2d(d_model*2, d_model, kernel_size=1, stride=1)

        self.sr_ratio = sr_ratio
        # 实现上这里等价于一个卷积层
        if sr_ratio > 1:
            self.sr_rgb = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_rgb = nn.LayerNorm(d_model)


        if sr_ratio > 1:
            self.sr_ir = nn.Conv2d(d_model, d_model, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm_ir = nn.LayerNorm(d_model)



        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        B, N ,C= x.shape

        h=int(math.sqrt(N//2))
        w=h


        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
     #   token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat

        x_rgb,x_ir =  torch.split(x,N//2,dim=1)	# 按单位长度切分，可以使用一个列表

        x = x_rgb
        if self.sr_ratio > 1:

            x_ = x.permute(0, 2, 1).reshape(B, C, h,w)    
            x_ = self.sr_rgb(x_).reshape(B, C, -1).permute(0, 2, 1) # 这里x_.shape = (B, N/R^2, C)
            x_ = self.norm_rgb(x_)
            kv_rgb = self.kv_rgb(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv_rgb = self.kv_rgb(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)



        x = x_ir
        if self.sr_ratio > 1:

            x_ = x.permute(0, 2, 1).reshape(B, C, h,w)    
            x_ = self.sr_ir(x_).reshape(B, C, -1).permute(0, 2, 1) # 这里x_.shape = (B, N/R^2, C)
            x_ = self.norm_ir(x_)
            kv_ir = self.kv_ir(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv_ir = self.kv_ir(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            
        k_rgb, v_rgb = kv_rgb[0], kv_rgb[1]
        k_ir, v_ir = kv_ir[0], kv_ir[1]

        attn_rgb = (q @ k_rgb.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb.softmax(dim=-1)
        attn_rgb = self.attn_drop_rgb(attn_rgb)
 


        attn_ir = (q @ k_ir.transpose(-2, -1)) * self.scale
        attn_ir = attn_ir.softmax(dim=-1)
        attn_ir = self.attn_drop_ir(attn_ir)
 

        x_rgb = (attn_rgb @ v_rgb).transpose(1, 2).reshape(B, N, C)
        x_rgb = self.proj_rgb(x_rgb)
        out_rgb = self.proj_drop_rgb(x_rgb)


        x_ir = (attn_ir @ v_ir).transpose(1, 2).reshape(B, N, C)
        x_ir = self.proj_ir(x_ir)
        out_ir = self.proj_drop_ir(x_ir)

        out_rgb_1,out_rgb_2 =  torch.split(out_rgb,N//2,dim=1)	# 按单位长度切分，可以使用一个列表
        out_ir_1,out_ir_2 =  torch.split(out_ir,N//2,dim=1)	# 按单位长度切分，可以使用一个列表

        out_rgb_1_ = out_rgb_1.permute(0, 2, 1).reshape(B, C, h,w)    
        out_rgb_2_ = out_rgb_2.permute(0, 2, 1).reshape(B, C, h,w)    

        out_ir_1_ = out_ir_1.permute(0, 2, 1).reshape(B, C, h,w)    
        out_ir_2_ = out_ir_2.permute(0, 2, 1).reshape(B, C, h,w)  

        out_rgb = self.out_rgb(torch.cat([out_rgb_1_, out_rgb_2_], dim=1) ) # concat
        out_ir = self.out_ir(torch.cat([out_ir_1_, out_ir_2_], dim=1))  # concat


        out_rgb=out_rgb.view(B, C, -1).permute(0, 2, 1)
        out_ir=out_ir.view(B, C, -1).permute(0, 2, 1)

        out = torch.cat([out_rgb, out_ir], dim=1)  # concat


        return out

class myTransformerCrossBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop,sr_ratio):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)

        self.sa = CrossAttention(d_model, d_k, d_v, h,sr_ratio, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        #mlp feng  jiang wei



        x = x + self.mlp(self.ln_output(x))

        return x


class GPTcross(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model,sr_ratio, vert_anchors, horz_anchors,n_layer, 
                    h=8, block_exp=4,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb1 = nn.Parameter(torch.zeros(1,  vert_anchors * horz_anchors, self.n_embd))
        self.pos_emb2 = nn.Parameter(torch.zeros(1,  vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerCrossBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop,sr_ratio)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool_rgb = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        self.avgpool_ir = nn.AdaptiveAvgPool2d((self.vert_anchors, self.horz_anchors))
        # self.DWconv = DWConv(d_model,d_model,kernel_size=d_model/self.vert_anchors,stride=d_model/self.vert_anchors)

        # 映射的方式

        # self.mapconv_rgb = nn.Conv2d(d_model *2, d_model, kernel_size=1, stride=1, padding=0)
        # self.mapconv_ir = nn.Conv2d(d_model *2, d_model, kernel_size=1, stride=1, padding=0)

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool_rgb(rgb_fea)
        ir_fea = self.avgpool_ir(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature


        # token_embeddings = torch.cat([rgb_fea_flat, ir_fea_flat], dim=2)  # concat

        rgb_token_embeddings = rgb_fea_flat
        ir_token_embeddings = ir_fea_flat


        rgb_token_embeddings = rgb_token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
        ir_token_embeddings = ir_token_embeddings.permute(0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        x_rgb = self.drop(self.pos_emb1 + rgb_token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)
        x_ir = self.drop(self.pos_emb2 + ir_token_embeddings)  # sum positional embedding and token    dim:(B, 2*H*W, C)

        x = torch.cat([x_rgb, x_ir], dim=1)  # concat
        
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        
        #映射的方式

        # all_fea_out = torch.cat([rgb_fea_out_map, ir_fea_out_map], dim=1)  # concat
        # rgb_fea_out = self.mapconv_rgb(all_fea_out)
        # ir_fea_out= self.mapconv_ir(all_fea_out)


        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out

def parse_CALNet_GPTcross_method(ch, f, n, m, args):
    c2 = ch[f[0]]
    args = [c2,args[1],args[2],args[3],args[4]]
    
    return ch[f[0]], ch[f[0]], n, args