import torch
import torch.autograd
import math
import torch.nn.functional as F

from einops import rearrange, einsum

import triton
import triton.language as tl

# from FlashAttention import flash_bwd
# compiled_bwd=torch.compile(flash_bwd)

@triton.jit
def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal:tl.constexpr
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D,),
        strides=(stride_qq, stride_qd,),
        offsets=(query_tile_index * Q_TILE_SIZE, 0,),
        block_shape=(Q_TILE_SIZE, D,),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd,),
        offsets=(0, 0),
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

    Q_tile = tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")
    O_acc = tl.zeros((Q_TILE_SIZE, D,), dtype=tl.float32)
    l_acc = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_acc = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    num_key_tiles = tl.cdiv(N_KEYS, K_TILE_SIZE)

    for j in range(num_key_tiles):
        flag=True        #flag来作为一个是否运行后面代码的指标
        if is_causal:
            query_end_pos=(query_tile_index+1)*Q_TILE_SIZE    #Q块完了之后的最后一行的下一行偏移
            key_start_pos=j*K_TILE_SIZE                       #K小块的起始偏移
            #如果Q块的结束位置下一个 是比K块起始偏移小（或等），那么就总有 横标小于纵标，从而整个块都可以定义为一个掩码值
            if key_start_pos>=query_end_pos:
                flag=False

        if flag:

            K_tile = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
            V_tile = tl.load(V_block_ptr,boundary_check=(0,1),padding_option="zero")

            S_tile=tl.zeros((Q_TILE_SIZE,K_TILE_SIZE), dtype=tl.float32)
            S_tile = tl.dot(Q_tile, tl.trans(K_tile),acc=S_tile) * scale

            #思考：1.仅对角处需要掩码的计算，对角线上方的可以直接跳过，而对角线下方直接计算非掩码即可；2.用了padding_option时，我们无需再去验证最后的一次j是否存在空值
            #第一个优化是无法在这里的下面实现的，因为triton内部在进入分支过后又会串行进行处理，而它的实现只能在一开始的时候就去判断,我们利用一个示性的flag来实现
            if is_causal:
                query_offset = query_tile_index * Q_TILE_SIZE + tl.arange(0,Q_TILE_SIZE)
                key_offset = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
                causal_mask = query_offset[:, None] >= key_offset[None, :]
                S_tile = tl.where(causal_mask, S_tile,S_tile-1e6)

            m_new = tl.maximum(m_acc, tl.max(S_tile, axis=-1))
            P_tile = tl.exp(S_tile - m_new[:, None])

            l_new = tl.exp(m_acc - m_new) * l_acc + tl.sum(P_tile,axis=-1)

            scale_factor = tl.exp(m_acc - m_new)
            O_acc=O_acc * scale_factor[:, None]
            O_acc = tl.dot(P_tile.to(V_tile.dtype), V_tile,acc=O_acc)

            m_acc = m_new
            l_acc = l_new



        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    O_final = O_acc / l_acc[:, None]
    L_final = m_acc + tl.log(l_acc)

    O_final = O_final.to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, O_final,boundary_check=(0,1))
    tl.store(L_block_ptr, L_final,boundary_check=(0,))

@triton.jit
def bwd_calculate_d_kernel(
        O_ptr,dO_ptr,
        D_ptr,
        stride_ob,stride_oq,stride_od,
        stride_dob,stride_doq,stride_dod,
        stride_db,stride_dq,
        N_QUERIES,
        D_MODEL:tl.constexpr,
        BLOCK_SIZE:tl.constexpr
):
    batch_idx=tl.program_id(0)
    query_tile_idx=tl.program_id(1)

    O_block_ptr=tl.make_block_ptr(
        O_ptr+batch_idx*stride_ob,
        shape=(N_QUERIES,D_MODEL,),
        strides=(stride_oq,stride_od,),
        offsets=(query_tile_idx*BLOCK_SIZE,0,),
        block_shape=(BLOCK_SIZE,D_MODEL,),
        order=(1,0),
    )
    dO_block_ptr=tl.make_block_ptr(
        dO_ptr+batch_idx*stride_dob,
        shape=(N_QUERIES,D_MODEL,),
        strides=(stride_doq,stride_dod,),
        offsets=(query_tile_idx*BLOCK_SIZE,0,),
        block_shape=(BLOCK_SIZE,D_MODEL,),
        order=(1,0),
    )
    O_row=tl.load(O_block_ptr,boundary_check=(0,),padding_option="zero")
    dO_row=tl.load(dO_block_ptr,boundary_check=(0,),padding_option="zero")

    D_row=tl.sum(O_row.to(tl.float32)*dO_row.to(tl.float32),axis=-1)    #形成形状为(BLOCK_SIZE,)的张量
    D_block_ptr=tl.make_block_ptr(
        D_ptr+batch_idx*stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_idx*BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,)
    )
    tl.store(D_block_ptr,D_row,boundary_check=(0,))

@triton.jit
def flash_bwd_kernel(
        Q_ptr,K_ptr,V_ptr
        #O_ptr
        ,L_ptr,dO_ptr,D_ptr,dQ_ptr,dK_ptr,dV_ptr,
        stride_qb,stride_qq,stride_qd,
        stride_kb,stride_kk,stride_kd,
        stride_vb,stride_vk,stride_vd,
        #stride_ob,stride_oq,stride_od,
        stride_lb,stride_lq,
        stride_dob,stride_doq,stride_dod,
        stride_db,stride_dq,
        stride_dqb,stride_dqq,stride_dqd,
        stride_dkb,stride_dkk,stride_dkd,
        stride_dvb,stride_dvk,stride_dvd,
        N_QUERIES,N_KEYS,
        scale,
        D:tl.constexpr,                                       #注意这个D是dim，而非计算的中间值D
        Q_TILE_SIZE:tl.constexpr,
        K_TILE_SIZE:tl.constexpr,
        is_causal:tl.constexpr
):
    key_tile_idx=tl.program_id(0)                             #注意反向传播的算法是外层为K，V循环，但是在这里我们会把循环调成并行
    batch_idx=tl.program_id(1)                                #先并行把K V的完成后 再调整批次

    K_block_ptr=tl.make_block_ptr(
        K_ptr+batch_idx*stride_kb,
        shape=(N_KEYS,D,),
        strides=(stride_kk,stride_kd,),
        offsets=(key_tile_idx*K_TILE_SIZE,0,),
        block_shape=(K_TILE_SIZE,D,),
        order=(1,0)
    )

    V_blcok_ptr=tl.make_block_ptr(
        V_ptr+batch_idx*stride_vb,
        shape=(N_KEYS,D,),
        strides=(stride_vk,stride_vd,),
        offsets=(key_tile_idx*K_TILE_SIZE,0,),
        block_shape=(K_TILE_SIZE,D,),
        order=(1,0)
    )

    K_tile=tl.load(K_block_ptr,boundary_check=(0,1),padding_option="zero")
    V_tile=tl.load(V_blcok_ptr,boundary_check=(0,1),padding_option="zero")

    dK_acc=tl.zeros((K_TILE_SIZE,D,),dtype=tl.float32)
    dV_acc=tl.zeros((K_TILE_SIZE,D,),dtype=tl.float32)

    Q_block_ptr=tl.make_block_ptr(
        Q_ptr+batch_idx*stride_qb,
        shape=(N_QUERIES,D,),
        strides=(stride_qq,stride_qd,),
        offsets=(0,0),                                 #每次都把偏移搞到矩阵左上角，我们每个线程是K V的固定整行，然后去遍历Q块，内层会进行advance，所以只需要在每个线程内至于原点即可
        block_shape=(Q_TILE_SIZE,D,),
        order=(1,0)
    )

    # O_block_ptr=tl.make_block_ptr(
    #     O_ptr+batch_idx*stride_ob,
    #     shape=(N_QUERIES,D,),
    #     strides=(stride_oq,stride_od,),
    #     offsets=(0,0),
    #     block_shape=(Q_TILE_SIZE,D,),
    #     order=(1,0)
    # )

    dO_block_ptr=tl.make_block_ptr(
        dO_ptr+batch_idx*stride_dob,
        shape=(N_QUERIES,D,),
        strides=(stride_doq,stride_dod,),
        offsets=(0,0),
        block_shape=(Q_TILE_SIZE,D,),
        order=(1,0)
    )

    # dQ_block_ptr=tl.make_block_ptr(
    #     dQ_ptr+batch_idx*stride_dqb,                之前设想的像forward一样的flag实现貌似有问题，我们现在正常掩码
    #     shape=(N_QUERIES,D,),
    #     strides=(stride_dqq,stride_dqd,),
    #     offsets=(0,0),
    #     block_shape=(Q_TILE_SIZE,D,),
    #     order=(1,0)
    # )

    L_block_ptr=tl.make_block_ptr(
        L_ptr+batch_idx*stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )
    D_block_ptr=tl.make_block_ptr(
        D_ptr+batch_idx*stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,)
    )


    num_query_tiles=tl.cdiv(N_QUERIES,Q_TILE_SIZE)      #这里内层循环是Q

    for i in range(num_query_tiles):
        Q_tile=tl.load(Q_block_ptr,boundary_check=(0,1),padding_option="zero")

        dO_tile=tl.load(dO_block_ptr,boundary_check=(0,1),padding_option="zero")
        L_tile=tl.load(L_block_ptr,boundary_check=(0,),padding_option="zero")
        D_tile=tl.load(D_block_ptr,boundary_check=(0,),padding_option="zero")

        S_tile = tl.dot(Q_tile, tl.trans(K_tile)) * scale
        P_tile = tl.exp(S_tile - L_tile[:, None])
        # dV_acc =dV_acc+tl.dot(tl.trans(P_tile), dO_tile.to(tl.float32))
        # dP_tile = tl.dot(dO_tile.to(tl.float32), tl.trans(V_tile).to(tl.float32))
        # dS_tile=P_tile.to(tl.float32)*(dP_tile-D_tile[:,None])

        if is_causal:
            query_offset=i*Q_TILE_SIZE+tl.arange(0,Q_TILE_SIZE)           #把掩码码处的值设为0来使得梯度也清空
            key_offset=key_tile_idx*K_TILE_SIZE+tl.arange(0,K_TILE_SIZE)
            causal_mask=query_offset[:,None]>=key_offset[None,:]
            P_tile=tl.where(causal_mask,P_tile,0.0)
            #dS_tile=tl.where(causal_mask,dS_tile,0.0)

        dV_acc = dV_acc + tl.dot(tl.trans(P_tile), dO_tile.to(tl.float32))
        dP_tile = tl.dot(dO_tile.to(tl.float32), tl.trans(V_tile).to(tl.float32))
        dS_tile = P_tile.to(tl.float32) * (dP_tile - D_tile[:, None])


        dS_tile_f32=dS_tile.to(tl.float32)
        dQ_i=tl.dot(dS_tile_f32,K_tile.to(tl.float32))*scale            #中间的加值

        dK_acc=dK_acc+tl.dot(tl.trans(dS_tile_f32),Q_tile.to(tl.float32))*scale

        query_offsets=i*Q_TILE_SIZE+tl.arange(0,Q_TILE_SIZE)
        dim_offsets=tl.arange(0,D)
        dQ_tile_ptr=dQ_ptr+batch_idx*stride_dqb+query_offsets[:,None]*stride_dqq+dim_offsets[None,:]*stride_dqd
        query_mask=query_offsets<N_QUERIES
        tl.atomic_add(dQ_tile_ptr,dQ_i,mask=query_mask[:,None])

        Q_block_ptr=Q_block_ptr.advance((Q_TILE_SIZE,0))
        dO_block_ptr=dO_block_ptr.advance((Q_TILE_SIZE,0))

        L_block_ptr=L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr=D_block_ptr.advance((Q_TILE_SIZE,))
    dK_block_ptr=tl.make_block_ptr(
        dK_ptr+batch_idx*stride_dkb,
        shape=(N_KEYS,D,),
        strides=(stride_dkk,stride_dkd,),
        offsets=(key_tile_idx*K_TILE_SIZE,0,),
        block_shape=(K_TILE_SIZE,D,),
        order=(1,0)
    )
    dV_block_ptr=tl.make_block_ptr(
        dV_ptr+batch_idx*stride_dvb,
        shape=(N_KEYS,D,),
        strides=(stride_dvk,stride_dvd,),
        offsets=(key_tile_idx*K_TILE_SIZE,0,),
        block_shape=(K_TILE_SIZE,D,),
        order=(1,0)
    )

    tl.store(dK_block_ptr,dK_acc.to(dK_block_ptr.type.element_ty),boundary_check=(0,1))
    tl.store(dV_block_ptr,dV_acc.to(dV_block_ptr.type.element_ty),boundary_check=(0,1))




class FlashAttention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False,scale=None):
        #思考：如下实现中仅考虑了3个维度的形式，对于增加了头维度的，我们记录头的维度，利用einops来变换为3个维度的；
        #另一方面，triton核是在每个线程上按列进行的，而此时处理2的幂次的张量效果更为显著，我们的实现中d上是一次性处理完的，我们对d进行2的幂次进一 o90
        original_shape=Q.shape
        d_model=original_shape[-1]
        if not scale:
            scale = 1.0 / math.sqrt(d_model)
        #2幂的进一
        next_pow_2_d=triton.next_power_of_2(d_model)
        flag_padding=(next_pow_2_d!=d_model)

        Q=rearrange(Q,"... n d -> (...) n d")
        K=rearrange(K,"... n d -> (...) n d")
        V=rearrange(V,"... n d -> (...) n d")

        if flag_padding:
            pad_width=next_pow_2_d-d_model
            Q=F.pad(Q,(0,pad_width))
            K=F.pad(K,(0,pad_width))
            V=F.pad(V,(0,pad_width))

        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        new_batch_size,n_queries,_=Q.shape
        _,n_keys,_=K.shape

        if flag_padding:
            O = torch.empty((new_batch_size,n_queries,d_model),device=Q.device,dtype=Q.dtype)
        else:
            O=torch.empty_like(Q)

        L = torch.empty((new_batch_size, n_queries,), device=Q.device, dtype=torch.float32)

        Q_TILE_SIZE = min(16, n_queries)
        K_TILE_SIZE = min(16, n_keys)

        num_query_tiles = triton.cdiv(n_queries, Q_TILE_SIZE)

        grid = (num_query_tiles, new_batch_size)

        flash_fwd_kernel[grid](
            Q, K, V, O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            n_queries, n_keys,
            scale,
            D=next_pow_2_d,
            Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal
        )
        #还原
        if flag_padding:
            O=O[...,:d_model]

        #使用original_shape来恢复
        O=O.reshape(*original_shape[:-2],*O.shape[1:])

        assert O.shape==original_shape
        ctx.is_causal = is_causal
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.scale = scale
        ctx.flag_padding=flag_padding
        ctx.next_pow_2_d=next_pow_2_d

        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        scale = ctx.scale
        is_causal = ctx.is_causal
        flag_padding = ctx.flag_padding
        next_pow_2_d=ctx.next_pow_2_d

        dO=grad_output.contiguous()
        d_model=Q.shape[-1]

        dO_3d = rearrange(dO, "... n d -> (...) n d")
        Q_3d=rearrange(Q,"... n d -> (...) n d")
        K_3d=rearrange(K,"... n d -> (...) n d")
        V_3d=rearrange(V,"... n d -> (...) n d")
        O_3d=rearrange(O,"... n d -> (...) n d")


        if flag_padding:
            pad_width=next_pow_2_d-d_model
            dO_3d=F.pad(dO_3d,(0,pad_width))
            Q_3d=F.pad(Q_3d,(0,pad_width))
            K_3d=F.pad(K_3d,(0,pad_width))
            V_3d=F.pad(V_3d,(0,pad_width))
            O_3d=F.pad(O_3d,(0,pad_width))

        grid_batch_size,n_queries,_=Q_3d.shape
        _,n_keys,_=K_3d.shape

        D=torch.empty_like(L)

        D_BLOCK_SIZE=min(16,n_queries)
        d_grid=(grid_batch_size,triton.cdiv(n_queries,D_BLOCK_SIZE))       #为计算D来并行

        bwd_calculate_d_kernel[d_grid](
            O_3d,dO_3d,D,
            O_3d.stride(0),O_3d.stride(1),O_3d.stride(2),
            dO_3d.stride(0),dO_3d.stride(1),dO_3d.stride(2),
            D.stride(0),D.stride(1),
            n_queries,D_MODEL=next_pow_2_d,BLOCK_SIZE=D_BLOCK_SIZE
        )

        dQ_3d=torch.zeros_like(Q_3d,dtype=torch.float32)
        dK_3d=torch.zeros_like(K_3d,dtype=torch.float32)
        dV_3d=torch.zeros_like(V_3d,dtype=torch.float32)

        Q_TILE_SIZE = min(16, n_queries)
        K_TILE_SIZE = min(16, n_keys)

        bwd_grid=(triton.cdiv(n_keys,K_TILE_SIZE),grid_batch_size)

        flash_bwd_kernel[bwd_grid](
            Q_3d,K_3d,V_3d,L,dO_3d,D,dQ_3d,dK_3d,dV_3d,
            Q_3d.stride(0), Q_3d.stride(1), Q_3d.stride(2),
            K_3d.stride(0), K_3d.stride(1), K_3d.stride(2),
            V_3d.stride(0), V_3d.stride(1), V_3d.stride(2),
            L.stride(0), L.stride(1),
            dO_3d.stride(0), dO_3d.stride(1), dO_3d.stride(2),
            D.stride(0), D.stride(1),
            dQ_3d.stride(0), dQ_3d.stride(1), dQ_3d.stride(2),
            dK_3d.stride(0), dK_3d.stride(1), dK_3d.stride(2),
            dV_3d.stride(0), dV_3d.stride(1), dV_3d.stride(2),
            n_queries,n_keys,scale,D=next_pow_2_d,Q_TILE_SIZE=Q_TILE_SIZE,K_TILE_SIZE=K_TILE_SIZE,is_causal=is_causal,
        )
        if flag_padding:
            dQ_3d=dQ_3d[...,:d_model]
            dK_3d=dK_3d[...,:d_model]
            dV_3d=dV_3d[...,:d_model]

        dQ=dQ_3d.reshape(*Q.shape[:-2],*dQ_3d.shape[1:]).to(Q.dtype)
        dK=dK_3d.reshape(*K.shape[:-2],*dK_3d.shape[1:]).to(K.dtype)
        dV=dV_3d.reshape(*V.shape[:-2],*dV_3d.shape[1:]).to(V.dtype)
        return dQ,dK,dV,None,None

#问题仍然存在。我们肉眼可见forward的掩码时候会耗时更少，但backward做不到这一点，可以再想想反向传播怎么像前向传播一样。