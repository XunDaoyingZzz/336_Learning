import torch
import torch.autograd
import math

import triton
import triton.language as tl

#compile用法见pdf第九页
def flash_bwd(Q,K,V,O,dO,L,scale,is_causal=False):
        D=torch.sum(O*dO,dim=-1)
        S=torch.einsum("... q d,... k d -> ... q k",Q,K)
        S=S*scale
        if is_causal:
            n_queries = Q.shape[-2]
            n_keys = K.shape[-2]
            # 创建一个上三角掩码
            mask = torch.triu(torch.ones(n_queries, n_keys, device=Q.device, dtype=torch.bool), diagonal=1)
            S.masked_fill_(mask, -float('inf'))
        P=torch.exp(S-L.unsqueeze(-1))
        dV=torch.einsum("... q k, ... q d -> ... k d",P,dO)
        dP=torch.einsum("... q d, ... k d -> ... q k",dO,V)
        dS=P*(dP-D.unsqueeze(-1))
        dQ=torch.einsum("... q k, ... k d -> ... q d",dS,K)
        dQ=dQ*scale
        dK=torch.einsum("... q k, ... q d -> ... k d",dS,Q)
        #dK=torch.matmul(dS.transpose(-2,-1),Q)
        dK=dK*scale
        return dQ,dK,dV


compiled_bwd=torch.compile(flash_bwd)

class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q,K,V,is_causal=False):
        original_Q_shape = Q.shape
        original_K_shape = K.shape
        original_V_shape = V.shape
        if len(Q.shape)==2:
            Q=Q.unsqueeze(0).unsqueeze(0)
            K=K.unsqueeze(0).unsqueeze(0)
            V=V.unsqueeze(0).unsqueeze(0)

        if len(K.shape)==3:
            Q=Q.unsqueeze(1)
            K=K.unsqueeze(1)
            V=V.unsqueeze(1)

        Q_shape_before,seq_len,d=Q.shape[:-2],Q.shape[-2],Q.shape[-1]
        B_q=min(16,seq_len)       #定义Q的块的行大小
        B_k=min(16,seq_len)       #定义KV的块的行大小，注意一件事K会进行转置，这个行分块转置后是裂分块

        T_q=math.ceil(seq_len/B_q)  #Q的分块数
        T_k=math.ceil(seq_len/B_k)  #K,V的分块数

        #对输出的O和维护的代理值l 进行空初始化
        O=torch.empty_like(Q)
        L=torch.empty((*Q_shape_before,seq_len),dtype=Q.dtype,device=Q.device)

        for i in range(T_q):
            q_start=i*B_q
            q_end=min((i+1)*B_q,seq_len)

            Q_i=Q[:,:,q_start:q_end,:]    #加载一个行分块
            O_i=torch.zeros(Q_i.shape,dtype=torch.float32,device=Q.device)     #小块的输出形状是(B,num_heads,B_q,d)
            l_i=torch.zeros((*Q_shape_before,q_end-q_start),dtype=torch.float32,device=Q.device)
            m_i=torch.full((*Q_shape_before,q_end-q_start),-float("inf"),dtype=torch.float32,device=Q.device)

            for j in range(T_k):
                kv_start=j*B_k
                kv_end=min((j+1)*B_k,seq_len)

                Kj=K[:,:,kv_start:kv_end,:]
                Vj=V[:,:,kv_start:kv_end,:]

                S_ij=torch.matmul(Q_i,Kj.transpose(-2,-1)/math.sqrt(d))

                m_ij=torch.max(S_ij,dim=-1)[0]   #维护这一行的最大值,形状成为(B,num_heads,B_q)
                m_i_new=torch.maximum(m_i,m_ij)

                P_ij=torch.exp(S_ij-m_i_new.unsqueeze(-1))           #前者(B,num_heads,B_q,B_k)，后者m_i_new变为(B,num_heads,B_q,1)，然后作广播减法，某一行全减去后者相同行的东西

                l_i_new=torch.exp(m_i-m_i_new)*l_i+torch.sum(P_ij,dim=-1) #形状(B,num_heads,B_q)
                scale_factor=torch.exp(m_i-m_i_new).unsqueeze(-1)   #算出exp自升一个维度 (B,num_heads,B_q,1)
                O_i=scale_factor*O_i +torch.matmul(P_ij,Vj)

                m_i=m_i_new
                l_i=l_i_new

            O_i=O_i/l_i.unsqueeze(-1)

            O[:,:,q_start:q_end,:]=O_i.to(Q.dtype)
            L[:,:,q_start:q_end]=m_i+torch.log(l_i)

        if len(original_Q_shape)==2:
            O=O.squeeze(0).squeeze(0)
            Q=Q.squeeze(0).squeeze(0)
            K=K.squeeze(0).squeeze(0)
            V=V.squeeze(0).squeeze(0)
            L=L.squeeze(0).squeeze(0)

        elif len(original_Q_shape)==3:
            O=O.squeeze(1)
            Q=Q.squeeze(1)
            K=K.squeeze(1)
            V=V.squeeze(1)
            L=L.squeeze(1)

        ctx.save_for_backward(L,Q,K,V,O)
        ctx.is_causal=is_causal
        return O

    @staticmethod
    def backward(ctx, grad_output):
        L,Q,K,V,O=ctx.saved_tensors
        d_model=Q.shape[-1]
        scale=1./math.sqrt(d_model)
        dQ,dK,dV=compiled_bwd(Q,K,V,O,grad_output,L,scale)

        return dQ, dK, dV,None

#刚才实施的那个纯torch版的可能还存在bug而且性能较差，下面编写triton核来并行实现。

@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,  # 输入QKV矩阵存储区域的指针
        O_ptr, L_ptr,  # 输出O L存储区域的指针
        stride_qb, stride_qq, stride_qd,  # 分别是q移动一个批次跳过元素的步长，查询个数维度上移动一个单位跳过元素的步长，以及特征维度移动一次的步长(这个通常为1)
        stride_kb, stride_kk, stride_kd,  # K的同上
        stride_vb, stride_vk, stride_vd,  # V的同上
        stride_ob, stride_oq, stride_od,  # O的同上
        stride_lb, stride_lq,  # L仅有两个维度，原因是L是某行指数和的对数，所以自然没有d维度
        N_QUERIES, N_KEYS,  # 查询数和键数的总值
        scale,  # 点积缩放因子
        D: tl.constexpr,  # 特征的总维度
        Q_TILE_SIZE: tl.constexpr,  # 查询分块在query上的大小B_q
        K_TILE_SIZE: tl.constexpr,  # 键分块在Key上的大小B_k
        is_causal:tl.constexpr
):
    query_tile_index = tl.program_id(0)  # 获取该线程的查询区块索引
    batch_index = tl.program_id(1)  # 获取该线程的批次索引

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,  # 经过批次索引找到当前批次的Q张量首指针
        shape=(N_QUERIES, D,),
        strides=(stride_qq, stride_qd,),
        offsets=(query_tile_index * Q_TILE_SIZE, 0,),
        block_shape=(Q_TILE_SIZE, D,),
        order=(1, 0),  # 这是默认序，事实上0是按行的轴，1是按列的轴
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd,),
        offsets=(0, 0),  # 也就是说在对Q分块进行矩阵乘法的时候，我们会重新遍历K
        block_shape=(K_TILE_SIZE, D,),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd,),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D,),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D,),
        strides=(stride_oq, stride_od,),
        offsets=(query_tile_index * Q_TILE_SIZE, 0,),
        block_shape=(Q_TILE_SIZE, D,),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_tile = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")  # 利用指针加载到当前对应的Q小片,我们在主进程中会并行加载多个Q_tile，然后每个tile逐次在若干行上向右移动，小块形状(Q_TILE_SIZE,D,)
    # 然后开始实现算法
    # 有个问题，我们在纯torch内部是利用了empty来实现了内存优化，但是triton内部不用这样做，因为每次仅加载小块；不仅如此，我们也不能这么做，kernel是纯粹的计算单元，不具备动态分配或释放全局内存的能力，也就是说占用的内存在一开始分配区块的时候就定死了。另外计算单元需要用float32提高精度
    O_acc = tl.zeros((Q_TILE_SIZE, D,), dtype=tl.float32)

    l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    m_acc = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)

    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)  # 算法中的T_k
    # 注意编写的时候我们没对Q的列再划分

    for j in range(num_key_tiles):

        K_tile = tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")  # 小块形状(K_TILE_SIZE,D,)
        V_tile = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")  # 小块形状(K_TILE_SIZE,D,)

        # 计算注意力得分 S_ij=Q_i@K_j^T/sqrt(d)->(Q_TILE_SIZE,K_TILE_SIZE)
        S_tile=tl.zeros((Q_TILE_SIZE,K_TILE_SIZE), dtype=tl.float32)
        S_tile = tl.dot(Q_tile, tl.trans(K_tile),acc=S_tile) * scale

        # 这里看是否需要因果掩码
        if is_causal:
            query_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)  # 计算查询的绝对位置，在arange里面自动广播了Q_TILE_SIZE个，形成当前块的所有行标
            key_offset = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # 键的绝对位置
            # 这里做个解释，前者升维到[Q_TILE_SIZE,1]后者到[1,K_TILE_SIZE]，前者的每一行元素和后者的每一列元素进行判断，形成[Q_TILE,K_TILE]矩阵，只有前者的当前元素更大 才为1，否则为0，形成下三角
            causal_mask = query_offset[:, None] >= key_offset[None, :]
            # 应用掩码
            S_tile = tl.where(causal_mask, S_tile,S_tile-1e6)  # 0处设置为负无穷，这样在softmax层算出来就是0了

        if j == num_key_tiles - 1:  # 循环来到了最后一次,注意前面我们的计算用了向上取整，最后一次迭代是存在无效元的
            key_mask = tl.arange(0, K_TILE_SIZE) < (N_KEYS - j * K_TILE_SIZE)  # 无效位置被标记为0
            key_mask = key_mask[None, :]  # 自升一维[1,K_TILE_SIZE]便于广播
            key_mask = tl.broadcast_to(key_mask, [Q_TILE_SIZE, K_TILE_SIZE])  # 广播成[Q_TILE_SIZE,K_TILE_SIZE]
            S_tile = tl.where(key_mask, S_tile, -float("inf"))  # 1的位置还是S_tile,0的位置标记为负无穷，这样取e后就是0，不占权重


        # 更新维护行内最大值：m_i^j=max(m_i^{j-1},rowmax(S_i^j))
        m_new = tl.maximum(m_acc, tl.max(S_tile, axis=-1))  # m_new是广播求的最大值，形状是(Q_TILE_SIZE,)每个元素是每一行的当前最大值
        P_tile = tl.exp(S_tile - m_new[:, None])  # 把m_new自升维到(Q_TILE_SIZE,1),然后自动广播S_tile每行减去相同的max值

        # 计算新的代理值：l_i^j=exp(m_i^{j-1}-m_i^{j})*l_ij+rowsum(P_i^{j}) 注意*是逐元素乘法
        l_new = tl.exp(m_acc - m_new) * l_acc + tl.sum(P_tile,axis=-1)  # 总之就是传统softmax我们仅用tl.sum一部分就够了，但是这里为了数值稳定我们额外加了一个tl.exp *l_acc来作为分母值

        # 更新输出：O_i^j=diag(exp(m_i^{j-1}-m_i^{j}))@O_i^{j-1}+P_tile@V_tile
        scale_factor = tl.exp(m_acc - m_new)
        O_acc=O_acc * scale_factor[:, None]
        O_acc = tl.dot(P_tile.to(V_tile.dtype), V_tile,acc=O_acc)

        m_acc = m_new
        l_acc = l_new
        # 迭代过程中需要移动的只有K，V块指针
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_final = O_acc / l_acc[:, None]  # 把l_acc变为[Q_TILE_SIZE,1]然后逐行相除
    # 计算Logsumexp L_i=m_i^{T_k}+log(l_i^{T_k})
    L_final = m_acc + tl.log(l_acc)

    O_final = O_final.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_final,boundary_check=(0,1))  # 存储指针，内容可由指针找到
    tl.store(L_block_ptr, L_final,boundary_check=(0,))

class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False,scale=None):
        batch_size, n_queries, d_model = Q.shape
        _, n_keys, _ = K.shape

        if not scale:
            scale = 1.0 / math.sqrt(d_model)

        # 确保张量是连续存储
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        O = torch.empty_like(Q)  # 现在不是triton核，可用空初始化
        L = torch.empty((batch_size, n_queries,), device=Q.device, dtype=torch.float32)

        # 块规模不能超过对应维度总长
        Q_TILE_SIZE = min(16, n_queries)
        K_TILE_SIZE = min(16, n_keys)

        # Q的线程分割
        num_query_tiles = triton.cdiv(n_queries, Q_TILE_SIZE)

        # 设置triton核函数的处理网格，第一维我们先在某个批次上处理分割的线程，第二位我们处理完某个批次的线程后移动批次来处理下一个批次
        grid = (num_query_tiles, batch_size)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=d_model,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )

        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        dQ, dK, dV = compiled_bwd(Q, K, V, O, grad_output, L, scale,is_causal)

        return dQ, dK, dV, None,None

#注意：目前的实现backward部分仍然是标准的直接载入，虽然采用了重计算，但是内存开销仍然不小，相对于forward的triton部分内存大的离谱。


