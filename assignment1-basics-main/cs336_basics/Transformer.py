from typing import Union, Callable, Any, Optional

import torch
#from scipy.signal import max_len_seq
from torch import nn
from torch.nn import init, factory_kwargs
from einops import rearrange, einsum
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class Linear(nn.Module):
    """自建的线性层"""
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.in_features=in_features
        self.out_features=out_features
        factory_kwargs={"device":device,"dtype":dtype}  #factory_kwargs存储设备信息和数字类型


        self.Weights=nn.Parameter(torch.empty((self.out_features,self.in_features),**factory_kwargs)) #建立空的权数矩阵
        std=(2/(in_features+out_features))**0.5
        init.trunc_normal_(self.Weights,mean=0.,std=std,a=-3*std,b=3*std) #四个参数第一个参数是要初始化的张量，第二个是均值，第三个是标准差，第四个是截断处

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return einsum(x,self.Weights,"... i,o i->... o")


class Embedding(nn.Module):
    """自建的嵌入层"""
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        self.num_embeddings=num_embeddings  #输入的时候的种类数
        self.embedding_dim=embedding_dim    #输出的时候的种类数
        factory_kwargs={"device":device,"dtype":dtype}

        self.Embedding_Matrix=nn.Parameter(torch.empty((self.num_embeddings,self.embedding_dim),**factory_kwargs))

        init.trunc_normal_(self.Embedding_Matrix,mean=0.,std=1,a=-3,b=3)

    def forward(self,token_ids:torch.LongTensor):  #解释一下，我们的Embedding矩阵是num_embeddings*embedding_dim的，是把原来的词表所有词元都给了个embedding_dim维的编码
        return self.Embedding_Matrix[token_ids]    #然后输入的样本token_ids 是(batch_size,sequence_length) 也就是说分了batch，每一个批次是length个词元，把这length个词元id去对应到嵌入矩阵的行标，然后把这些行拼成的矩阵取出来拼成输出的矩阵.



class RMS_Norm(nn.Module):
    """自建的均方层归一化"""
    def __init__(self,d_model:int,eps:float=1e-5,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        factory_kwargs={"device":device,"dtype":dtype}
        self.gain=nn.Parameter(torch.ones(d_model,**factory_kwargs))  #搞一个可以训练的参数来使得模型达到更好的归一化状态

    def forward(self,x:torch.Tensor)->torch.Tensor:
        in_dtype=x.dtype      #记录类型
        x=x.to(torch.float32) #把x转化为float32型体高精度
        rms=torch.square(x)   #先自平方
        rms=torch.sqrt(rms.mean(dim=-1,keepdim=True)+self.eps) #平方平均
        x=x/rms*self.gain     #归一化
        return x.to(in_dtype) #还原类型

def SiLU(x:torch.Tensor):      #silu就是\frac{x}{1+e^{-x}}
    return x*torch.sigmoid(x)

def GLU(x:torch.Tensor,W_1,W_2):  #GLU操作
    return torch.sigmoid(W_1@x)*(W_2@x) #对x分别进行两个线性变换，其中一个做sigmoid，然后逐元素相乘

def SwiGLU(x,W_1,W_2,W_3):
    return W_2@(SiLU(W_1@x)*(W_3@x))  #先分别对x进行两个线性变换，然后一份做silu，并且逐元素相乘后对结果再进行一次线性变换

class positionwise_feedforward(nn.Module):
    """位置前馈神经网络"""
    def __init__(self,d_model,dff=None,device=None,dtype=None):
        super().__init__()
        self.d_model=d_model
        if dff:
            self.dff=dff
        else:
            s=d_model*8/3 #没有传入dff的话，按模型dim的8/3来作为dff，但是由于我们期望是2的倍数，会在下面进行一个细致的2幂变换
            self.dff=int(((s+32)//64)*64)  #实现一个64位的四舍五入
        self.SiLU=SiLU 
        factory_kwargs={"device":device,"dtype":dtype}
        self.W_1=Linear(self.d_model,self.dff,**factory_kwargs) #d_model->dff的线性层
        self.W_2=Linear(self.dff,self.d_model,**factory_kwargs) #dff->d_model的线性层
        self.W_3=Linear(self.d_model,self.dff,**factory_kwargs) #d_model->dff的线性层

    def forward(self,x):
        return self.W_2(self.SiLU(self.W_1(x))*self.W_3(x))

class RotaryPositionalEmbedding(nn.Module):
    """位置编码嵌入"""
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):#theta是基数，d_k是键(k)与查询(v)的维度，max_seq_len是期望的最大序列长度
        super().__init__()
        if d_k%2!=0:
            raise ValueError("我们需要偶数的维度")  #详情见RoPE的推导，换句话说我们会取一部分作为sin一部分作为cos
        self.d_k=d_k    #如果d_k=2m，下面我们就可以取到i=0->m-1,1/theta^{i/m}的一列数
        freq=1.0/(theta**(torch.arange(0,d_k//2,device=device).float()/(d_k//2)))

        positions=torch.arange(max_seq_len,device=device).float() #序列长度是行数，生成遍历行数的一列
        freqs=torch.outer(positions,freq)   #outer计算外积，1个是a维的向量，一个是b维的向量，放一起就是(a,b)的tensor
        #与其说outer是外积，不如说它是一个a*1 与1*b的矩阵乘法形成的a*b矩阵，这样每一行是某次序列的某一既定词元的嵌入信息，反应的是相同位置的不同嵌入信息
        #同时每一列是某一个嵌入标签下不同词元位置的信息

        #register_buffer注册为缓冲区，会随模型移动设备但是不参与梯度计算；计算所有位置和频率组合的cos与sin，并且不保存到state_dict中
        self.register_buffer("cos_cached",torch.cos(freqs),persistent=False)
        self.register_buffer('sin_cached',torch.sin(freqs),persistent=False)

    def forward(self,x:torch.Tensor            #(... seq_len d_k)
                ,token_positions:torch.Tensor  #(... seq_len)
                )->torch.Tensor:
        if x.size(-1)!=self.d_k:  #如果最后一个维度上x的长度和d_k不匹配，我们报错,注意最后一个维度上的是某一既定词元的嵌入信息，反映的是相同位置的不同嵌入标签
            raise ValueError(f"x的长度与d_k要相当")

        #新加的debug
        *batch_dims,seq_len,d_k=x.shape #我们把seq_len和d_k取出来，并把前面的形状全部搞给batch_dims（他现在是一个维度列表）

        if len(token_positions.shape)<len(batch_dims)+1:   #如果token_position的维度是不比刚刚的batch_dims大的
            for _ in range(len(batch_dims)-len(token_positions.shape)+1): #我们需要的是token_positions的维度与batch_dims+1相匹配
                token_positions=token_positions.unsqueeze(-2)     

        #pos_shape=token_positions.shape
        #seq_len=pos_shape[-1]
        #batch_shape=pos_shape[:-1]

        #flat_positions=token_positions.contiguous().view(-1)
        #我们算出 cos (freq)  sin (freq)
        cos_pos=self.cos_cached[token_positions]
        sin_pos=self.sin_cached[token_positions]

        #target_shape=batch_shape+(seq_len,self.d_k//2)
        #cos_pos=cos_pos.view(target_shape)
        #sin_pos=sin_pos.view(target_shape)

        #cos_pos=self.cos_cached[token_positions]   #取出这些位置的位置编码
        #sin_pos=self.sin_cached[token_positions]

        x_even=x[...,::2]   #偶数位置处的切片 0，2，……2d_k-2
        x_odd=x[...,1::2]   #奇数位置处的切片 1，3，……2d_k-1
        #准备旋转 [x_even']=[cos -sin][x_even]
        #       [x_odd' ]=[sin  cos][x_odd  ]
        out_even=x_even*cos_pos-x_odd*sin_pos#0，2，……2d_k-2和cos (1/theta^{\frac{0}{m}}) cos (2/theta^{\frac{1}{m}})...
        out_odd=x_even*sin_pos+x_odd*cos_pos
        #然后交错排列
        out=torch.empty_like(x)  #未定义值的与x形状相同的张量
        out[...,::2]=out_even
        out[...,1::2]=out_odd
        return out

def softmax_stable(x,dim:int=-1):
    x_max=x.max(dim=dim,keepdim=True).values  #先找到最后这个维度上的最大值，并保留其他维度不变
    x_exp=torch.exp(x-x_max)  #防止x太大造成溢出
    return x_exp/x_exp.sum(dim=dim,keepdim=True)


class ScaledDotProductAttention(nn.Module):  #qkv缩放点积注意力，这里注意昨业的式子存在问题，应该是对K做转置
    def __init__(self,d_k:int):
        super().__init__()
        self.scale=1.0/d_k**0.5

    def forward(self,query,key,value,mask=None):
        attn_scores=einsum(query,key,"... q d,... k d-> ... q k")*self.scale

        if mask is not None:
            attn_scores=attn_scores.masked_fill(~mask.bool(),float("-inf"))   #~是取反，我们把含0的取为1，然后对所有的1处进行一个负无穷替换，这样e^-inf就是0

        attn_probs=softmax_stable(attn_scores,dim=-1)

        output=einsum(attn_probs,value,"... q k,... k d->... q d") #这样最后两层是q*v的矩阵，某一行上的v个元素是这一行的键取到各个值的权数值，某一列是某个值在各个键下的加权值
        return output


class CausalMultiHeadSelfAttention(nn.Module):   #因果多头自注意力机制
    def __init__(self,d_model:int,num_heads:int,max_seq_len:int,rope_theta:float=10000.,use_rope:bool=True,device=None,dtype=None):
        super().__init__()
        assert d_model%num_heads==0  #我们想设置d_k=d_v=d_model/h
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.d_v=self.d_k
        self.use_rope=use_rope

        factory_kwargs={"device":device,"dtype":dtype}
        self.q_proj,self.k_proj,self.v_proj,self.o_proj=[Linear(d_model,d_model,**factory_kwargs) for _ in range(4)]  #q k v o都是同维度的线性变换
        self.attn=ScaledDotProductAttention(self.d_k)   #attn采用缩放点积注意力
        #下面的mask相当于是一个上三角矩阵
        mask=torch.tril(torch.ones(max_seq_len,max_seq_len,dtype=torch.bool,device=device))#.tril函数原型传入(input,diagonal=0,*,out=None)默认保留主对角线以上的元素(diagonal=0,>0是保留上主对角线上方第k条对角线及以上，<0相反)，out是是否输出未保留的可选bool项

        self.register_buffer('causal_mask',mask.unsqueeze(0).unsqueeze(0),persistent=False) #persistent=False说明这个缓冲区不会被包含在state_dict中对mask矩阵自加两个维度 (1,1,S,S)

        if use_rope:
            self.rope=RotaryPositionalEmbedding(theta=rope_theta,d_k=self.d_k,max_seq_len=max_seq_len,device=device)

    def forward(self,x,token_positions=None,):
        """传入的x是 b seq_len d,token_positions是默认为空"""
        B,S,_=x.shape  #把批次和序列长度传给B,S
        #对于下面的qkv做下说明，对x做三次线性变换转换成查询 键 值 并做维度调整，最后两个维度调整为seq和d_k方便后续运算
        q,k,v=[rearrange(proj(x),"b s (h d)-> b h s d",h=self.num_heads) for proj in [self.q_proj,self.k_proj,self.v_proj]]

        if self.use_rope:
            if token_positions is None:
                # 根据序列长度 S，创建从 0 到 S-1 的位置张量
                token_positions = torch.arange(S, device=x.device)
            q,k=self.rope(q,token_positions),self.rope(k,token_positions)  #对qk进行旋转位置编码，这样就有了位置信息


        out=self.attn(q,k,v,mask=self.causal_mask[...,:S,:S])  #计算掩码注意力，我们在最后两个维度仅取前S行和前S列，并且掩码的阵是上三角形的，计算会使attn也是个上三角形的
        out=rearrange(out,"b h s d -> b s (h d)")
        return self.o_proj(out)

class transformer_block(nn.Module):      #transfromer块
    """
    x -> RMSNorm->MHA->self+x->y -> RMSNorm->FF ->self+y->out
    """
    def __init__(self,d_model:int,num_heads:int,d_ff:int,max_seq_len:int,rope_theta:float=10000.,use_rope:bool=True,device=None,dtype=None):
        super().__init__()
        kwargs={"device":device,"dtype":dtype}

        self.norm1=RMS_Norm(d_model,**kwargs)
        self.attn=CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
            use_rope=use_rope,
            **kwargs
        )

        self.norm2=RMS_Norm(d_model,**kwargs)

        self.ff=positionwise_feedforward(d_model=d_model,dff=d_ff,**kwargs)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor|None=None)->torch.Tensor:
        b,s,_=x.shape

        attn_out=self.attn(self.norm1(x),token_positions=token_positions)
        x=x+attn_out  #残差链接

        ff_out=self.ff(self.norm2(x))
        x=x+ff_out        #残差链接
        return x

"""在这之前我们先做一个复制参数的函数"""
def _copy_param(target:torch.Tensor,source:torch.Tensor)->None: #传入源数据和目标数据，如果
    if source.shape==target.shape:
        target.data.copy_(source)
    elif source.T.shape==target.shape:
        target.data.copy_(source.T)
    else:
        raise ValueError(f"对不上宝贝")

class transformer_lm(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 context_length:int,
                 num_layers:int,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 rope_theta:float,
                 device=None,
                 dtype=None
                 ):
        super().__init__()
        kwargs={"device":device,"dtype":dtype}
        self.token_embedding=Embedding(vocab_size,d_model,**kwargs) #token_embedding 定义为输入vocab_size词汇表大小，把每个词汇嵌入到d_model维度上的层
        self.blocks=nn.ModuleList([
            transformer_block(d_model=d_model,num_heads=num_heads,d_ff=d_ff,max_seq_len=context_length,rope_theta=rope_theta,use_rope=True,**kwargs)
            for _ in range(num_layers)
        ])

        self.ln_final=RMS_Norm(d_model,**kwargs)         #最后的层归一
        self.lm_head=Linear(d_model,vocab_size,**kwargs) #

        self.context_length=context_length

    def forward(self,token_ids:torch.Tensor)->torch.Tensor:
        b,s=token_ids.shape     #批次 序列长，思考一下这里是否会出现bug，因为我们不知道传入的是否是2个维度的
        if s>self.context_length:
            raise ValueError(f"seq_len{s}太长了")

        x=self.token_embedding(token_ids)

        pos=torch.arange(s,device=token_ids.device).unsqueeze(0).expand(b,s) #创建0->s-1的张量，并自己提升一个维度，在多的维度上复制b次

        for blk in self.blocks:
            x=blk(x,token_positions=pos)

        x=self.ln_final(x)

        logits=self.lm_head(x) #b s vocab_size

        return logits

"""损失环节"""
def log_softmax_stable(z,dim=-1):
    max_z,_=torch.max(z,dim=dim,keepdim=True) #先找到一行里面最大的z
    z=z-max_z #每个都减去最大的,现在的是 z_i-max{z}
    #我们是要计算 log softmax(z),拆一下是 z_i-log(sum (e^z))
    #由于softmax(z)=softmax(z-max{z}),log又是z_i-max{z}-log(sum(e^{z-max{z}}))
    return z-torch.log(torch.sum(torch.exp(z),dim=dim,keepdim=True))

##注意，这里输入的是logit值，然后targets是下一个真实标签的索引（词汇表的id）
class cross_entropy():
    def __init__(self,inputs,targets):
        self.inputs=inputs
        self.targets=targets
        self.vocab_size=self.inputs.shape[-1]

    def forward(self):
        flat_inputs=self.inputs.view(-1,self.vocab_size)
        flat_targets=self.targets.view(-1)

        prob=log_softmax_stable(flat_inputs)
        row_indices=torch.arange(flat_inputs.shape[0],device=self.inputs.device)
        prob=prob[row_indices,flat_targets]
        loss=-torch.mean(prob)
        return loss





















