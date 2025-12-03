from collections.abc import Callable,Iterable
from typing import Optional
import torch
import math

class adamw(torch.optim.Optimizer):
    def __init__(self,params,betas=(0.9,0.999),lr=0.01,eps=1e-8,weight_decay=1e-2):
        if lr <0.0 or eps<0.0 or betas[0]<0.0 or betas[1]<0.0 or betas[0]>=1.0 or betas[1]>=1.0 or weight_decay<0.0:
            raise ValueError("超参数错误")


        default={"alpha":lr,"beta_1":betas[0],"beta_2":betas[1],"lambda_":weight_decay,"epsilon":eps}
        super().__init__(params,default) #把params标准化成self.param_groups：list，每个元素是参数组的字典，且带上了默认的超参数键值对

    def step(self,closure:Optional[Callable]=None):
        """
        closure是可选的闭包,默认情况下是None
        """
        loss=None

        if closure is not None:
            loss=closure()

        for group in self.param_groups:    #遍历__init__返回的self.param_groups
            for param in group["params"]:  #对每个参数组里面再遍历该组的每个参数张量
                with torch.no_grad():
                    grad=param.grad


                if grad.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度")

                state=self.state[param]  #每个参数都有一个状态字典，用于保存一阶二阶的动量等随训练演化的量

                if len(state)==0:        #如果没有状态
                    state["step"]=0      #那么现在就是第0步
                    state["exp_avg"]=torch.zeros_like(param.data)      #这是一阶动量m_t
                    state["exp_avg_sq"]=torch.zeros_like(param.data)   #这是二阶动量v_t

                exp_avg,exp_avg_sq=state["exp_avg"],state['exp_avg_sq'] #解包我们的m_t，v_t
                beta_1,beta_2=group["beta_1"],group["beta_2"]           #取出我们的beta

                state["step"]+=1 #步数自加

                param.data.mul_(1-group["alpha"]*group["lambda_"])      #这一步是解耦权衰 theta_t=theta_{t-1}*(1-alpha*lambda)
                    #以下g_t是梯度
                exp_avg.mul_(beta_1).add_(grad,alpha=1-beta_1)          #这一步是 m_t=beta_1*m_{t-1}+(1-beta_1)*g_t
                exp_avg_sq.mul_(beta_2).addcmul_(grad,grad,value=1-beta_2) #这一步是 v_t=beta_2*v_{t-1}+(1-beta_2)*(g_t^2)注意这里是元素级别平方
                    #早期修正
                bias_correction1=1-beta_1**state["step"] #m_hat=m_t/(1-beta_1^t)
                bias_correction2=1-beta_2**state["step"] #v_hat=v_t/(1-beta_2^t)

                step_size=group["alpha"]/bias_correction1 #有效步长(修正学习率)alpha/(1-beta_1^t)
                bias_correction2_sqrt=math.sqrt(bias_correction2)#获取sqrt{1-beta_2^t},与上一个修正学习率乘起来做alpha_t

                #input.addcdiv(tensor1,tensor2,*,value=1,out=None)->Tensor  ,value是可选的乘数，我们该算法选择-1是因为要减去后面那部分
                #output=input+value*\frac{tensor1}{tensor2} 其中是元素级别运算
                param.data.addcdiv_(exp_avg, #tensor1 是m_t
                                    exp_avg_sq.sqrt().add_(group['epsilon']).div_(bias_correction2_sqrt),
                                    value=-step_size)
                #其实我们在最早做了theta_t=theta_{t-1}-alpha*lambda*theta_{t-1}，刚才这一步就是再减去一个alpha_t*\frac{m}{sqrt{v_t}+epsilon}
        return loss

#余弦退火
def learning_rate_schedule(t,alpha_max,alpha_min,T_w,T_c):
    if t<T_w:
        alpha_t=t*alpha_max/T_w
    elif T_w<=t<=T_c:
        alpha_t=alpha_min+0.5*(1+math.cos(math.pi*(t-T_w)/(T_c-T_w)))*(alpha_max-alpha_min)
    else:
        alpha_t=alpha_min
    return alpha_t

#梯度裁剪
def gradient_clipping(parameters:Iterable,M,eps=1e-6):
    total_norm=0.0
    for param in parameters:
        if param.grad is not None:
            param_norm=param.grad.data.norm(2)
            total_norm+=param_norm**2

    total_norm=total_norm**0.5
    if total_norm>=M:
        clip_coef = M / (total_norm + eps)
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    return total_norm


