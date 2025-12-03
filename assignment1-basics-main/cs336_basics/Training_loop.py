import numpy as np
import math
import torch
from einops import rearrange,einsum
from torch.optim import Optimizer
from torch import nn
import typing
import os

def data_loading(x:np.ndarray,batch_size:int,context_length:int,device=None):
    max_start=len(x)-context_length #不-1是因为下一步右边是开的，下一步最右边最多取到max_start-1
    starts=np.random.randint(0,max_start,batch_size) #从0->max_start找batch_size个起点

    input_token=[x[start:start+context_length] for start in starts]
    next_token=[x[start+1:start+context_length+1] for start in starts]

    input_array=np.array(input_token,dtype=np.int64)
    next_array=np.array(next_token,dtype=np.int64)


    inputs=torch.from_numpy(input_array).to(device=device)
    targets=torch.from_numpy(next_array).to(device=device)

    return inputs, targets
#数据集过大有个优化办法，我们暂时留个白


#state_dict(self,*args,destination=None,prefix:str=...,keep_vars:bool=...,)->dict[str,Any]
#destination是用于储存结果的字典，如果是None则会建立一个新的OrderedDict，否则直接往字典里面写入参数和buffer
#prefix是用于给参数名加前缀的，prefix+name+"."来区分 类层
#keep_vars为True时返回的是torch.nn.Parameter或者buffer原本形式（带grad），False的时候就是detach()过的无梯度tensor

def save_checkpoint(model:nn.Module,
                    optimizer:torch.optim.Optimizer,
                    iteration:int,
                    out:str|os.PathLike|typing.BinaryIO):
    """
    把前三个参数中的所有状态信息保存到类文件的对象out中，对模型和优化器用state_dict方法来获取他们的状态
    用torch.save(obj,out)来保存obj到out路径，建议obj是字典格式，但是只要后续能加载检查点，用其他格式也行
    model;optimizer;iteration;out
    """
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    checkpoint_dict={
        'model_state_dict':model_state_dict,
        'optimizer_state_dict':optimizer_state_dict,
        'iteration':iteration
    }
    torch.save(checkpoint_dict,out)

def load_checkpoint(src:str|os.PathLike,model,optimizer):
    """
    从src（路径或者文件）中加载检查点，然后恢复模型和优化器，函数返回保存到检查点的迭代次数，使用torch.load(src)恢复内容
    并使用module和optimizer中的load_state_dict()来恢复到之前状态
    """
    checkpoint = torch.load(src)
    model_state_dict = checkpoint['model_state_dict']
    optimizer_state_dict = checkpoint['optimizer_state_dict']
    iteration = checkpoint['iteration']
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    return iteration


