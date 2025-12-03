import os
import sys
import time
import argparse  # 用于解析命令行参数，方便地配置超参数
import numpy as np
import torch



MODULE_PATH = "D://336_Learning//assignment1-basics-main//cs336_basics"
if MODULE_PATH not in sys.path:
    sys.path.append(MODULE_PATH)

from Tokenization2 import BPETokenizer
from Transformer import transformer_lm, cross_entropy
from My_Optimizer import adamw, learning_rate_schedule, gradient_clipping
from Training_loop import data_loading, save_checkpoint, load_checkpoint


def pretokenize_and_memmap(text_path: str, tokenizer: BPETokenizer, save_dir: str):
    """
    此函数负责将原始文本文件预分词，并将生成的token ID序列保存到一个二进制文件中。
    这个二进制文件可以使用内存映射（memory-map）技术加载，从而实现高效的大文件处理。
    如果二进制文件已经存在，函数会直接加载它，避免重复分词。
    """
    # 如果用于存放预分词数据的缓存目录不存在，则创建它
    os.makedirs(save_dir, exist_ok=True)

    # 根据原始文本文件名生成二进制缓存文件的路径
    file_name = os.path.basename(text_path).replace('.txt', '.bin')
    bin_path = os.path.join(save_dir, file_name)

    # 检查缓存文件是否已经存在
    if os.path.exists(bin_path):
        print(f"找到已存在的预分词数据 {bin_path}，使用内存映射加载。")
        # 如果文件存在，使用np.memmap以只读模式('r')打开它。
        # 这种方式不会将整个文件加载到RAM中，而是像访问数组一样访问磁盘上的数据。
        # dtype=np.uint16是因为词汇表大小通常小于65535，使用16位无符号整数可以节省空间。
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    else:
        # 如果缓存文件不存在
        print(f"正在从 {text_path} 进行预分词...")
        # 读取整个文本文件
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # 使用你的BPE分词器对文本进行编码，得到token ID列表
        tokens = tokenizer.encode(text)
        # 将token ID列表转换为NumPy数组
        token_array = np.array(tokens, dtype=np.uint16)

        print(f"正在将分词结果保存到 {bin_path}...")
        # 以可写模式('w+')创建一个内存映射文件，形状与token_array相同
        data = np.memmap(bin_path, dtype=np.uint16, mode='w+', shape=token_array.shape)
        # 将token数据写入这个内存映射文件
        data[:] = token_array[:]
        # 确保所有数据都已写入磁盘
        data.flush()
        # 为了训练，我们以只读模式('r')重新打开它
        data = np.memmap(bin_path, dtype=np.uint16, mode='r')

    # 返回内存映射的数组
    return data


@torch.no_grad()  #装饰器，表示函数内的所有PyTorch操作都不会计算梯度。这在评估阶段非常重要，可以节省大量内存和计算资源。
def evaluate(model, data, context_length, batch_size, device, eval_iters=100):
    """
    在验证集上评估模型的性能（计算平均损失）。
    """
    model.eval()  # 将模型切换到评估模式。
    losses = []  # 用于存储每个批次的损失值
    # 循环指定的评估迭代次数
    for _ in range(eval_iters):
        # 加载一个批次
        inputs, targets = data_loading(data, batch_size, context_length, device)  #
        # 模型前向传播，得到预测的logits
        logits = model(inputs)

        # 计算损失
        loss_computer = cross_entropy(logits, targets)
        loss = loss_computer.forward()

        # 将当前批次的损失添加到列表中
        losses.append(loss.item())

    model.train()  # 评估结束后，将模型切换回训练模式
    return np.mean(losses)  # 返回所有评估批次的平均损失


def main(config):
    # 检查是否有可用的CUDA设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print("正在加载分词器...")
    # 从指定路径加载已经训练好的BPE分词器
    tokenizer = BPETokenizer.from_files(config.tokenizer_path)  #
    # 获取词汇表的大小，这将作为模型嵌入层和输出层的维度
    vocab_size = tokenizer.vocab_size
    print(f"分词器加载完成，词汇表大小: {vocab_size}")

    print("正在准备数据集...")
    # 使用预分词和内存映射函数处理训练集和验证集
    train_data = pretokenize_and_memmap(config.train_data_path, tokenizer, config.data_cache_dir)
    valid_data = pretokenize_and_memmap(config.valid_data_path, tokenizer, config.data_cache_dir)
    print(f"训练集包含 {len(train_data)} 个tokens。")
    print(f"验证集包含 {len(valid_data)} 个tokens。")


    print("正在初始化模型...")
    # 创建一个字典来存放所有模型相关的超参数
    model_args = {
        'vocab_size': vocab_size,
        'context_length': config.context_length,
        'num_layers': config.num_layers,
        'd_model': config.d_model,
        'num_heads': config.num_heads,
        'd_ff': config.d_ff,
        'rope_theta': config.rope_theta,
        'device': device,
        'dtype': torch.float32  # 可以根据GPU支持情况选择 torch.bfloat16 来加速
    }
    # 使用字典解包的方式(**model_args)来实例化的Transformer
    model = transformer_lm(**model_args).to(device)
    # 打印模型的参数量
    print(f"模型创建完成，总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")


    print("正在初始化优化器...")
    # 实例化自定义的AdamW优化器
    optimizer = adamw(model.parameters(), lr=config.lr_max, betas=tuple(config.betas),
                      weight_decay=config.weight_decay)  #

    # 检查点加载 (用于恢复训练)
    start_iter = 0  # 默认从第0次迭代开始
    # 如果命令行参数指定了要恢复的检查点文件
    if config.resume_from:
        print(f"正在从检查点恢复训练: {config.resume_from}")
        # 调用加载函数，它会更新模型和优化器的状态，并返回上次保存时的迭代次数
        start_iter = load_checkpoint(config.resume_from, model, optimizer)  #
        print(f"已从第 {start_iter} 次迭代恢复。")

    print("开始训练循环...")
    model.train()  # 确保模型处于训练模式
    start_time = time.time()  # 记录开始时间，用于计算迭代速度

    # 从start_iter开始循环，直到达到最大迭代次数
    for iter_num in range(start_iter, config.max_iters):
        #学习率调度 (使用带预热的余弦退火)
        lr = learning_rate_schedule(  #
            t=iter_num,
            alpha_max=config.lr_max,
            alpha_min=config.lr_min,
            T_w=config.warmup_iters,
            T_c=config.max_iters
        )
        # 将计算出的新学习率应用到优化器的每个参数组
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        #获取一个批次的数据
        inputs, targets = data_loading(train_data, config.batch_size, config.context_length, device)  #

        # 前向传播：模型根据输入计算预测结果(logits)
        logits = model(inputs)

        #计算损失
        loss_computer = cross_entropy(logits, targets)
        loss = loss_computer.forward()

        # 反向传播和优化
        optimizer.zero_grad(set_to_none=True)  # 清空之前的梯度。set_to_none=True是Pytorch推荐的优化方式，比赋0更高效。
        loss.backward()  # 计算损失相对于模型参数的梯度。

        #梯度裁剪：防止梯度爆炸，稳定训练过程
        if config.grad_clip > 0:
            gradient_clipping(model.parameters(), config.grad_clip)  #

        optimizer.step()  # 根据梯度更新模型的参数。

        # 定期日志、评估和保存
        # 每隔log_interval次迭代，打印一次训练日志
        if iter_num % config.log_interval == 0:
            elapsed_time = time.time() - start_time  # 计算逝去的时间
            print(f"迭代 {iter_num:6d} | 训练损失: {loss.item():.4f} | 学习率: {lr:.6f} | 耗时: {elapsed_time:.2f}s")
            start_time = time.time()  # 重置计时器

        # 每隔eval_interval次迭代，在验证集上进行一次评估
        if iter_num > 0 and iter_num % config.eval_interval == 0:
            val_loss = evaluate(model, valid_data, config.context_length, config.batch_size, device)
            print("-" * 50)
            print(f"迭代 {iter_num:6d} | 验证损失: {val_loss:.4f}")
            print("-" * 50)

        # 每隔save_interval次迭代，保存一次模型检查点
        if iter_num > 0 and iter_num % config.save_interval == 0:
            os.makedirs(config.checkpoint_dir, exist_ok=True)  # 确保检查点目录存在
            checkpoint_path = os.path.join(config.checkpoint_dir, f'ckpt_iter_{iter_num}.pt')
            print(f"正在保存检查点到 {checkpoint_path}")
            save_checkpoint(model, optimizer, iter_num, checkpoint_path)  #

    print("训练完成。")


if __name__ == '__main__':
    # 创建一个参数解析器
    parser = argparse.ArgumentParser(description="训练一个Transformer语言模型")

    # --- 路径和目录参数 ---
    parser.add_argument('--tokenizer_path', type=str, default='D://336_Learning//My_new_BPE2',
                        help='已保存的BPE分词器文件路径。')
    parser.add_argument('--train_data_path', type=str, default='D://336_Learning//TinyStoriesV2_GPT4-train.txt',
                        help='训练集文本文件路径。')
    parser.add_argument('--valid_data_path', type=str, default='D://336_Learning//TinyStoriesV2_GPT4-valid.txt',
                        help='验证集文本文件路径。')
    parser.add_argument('--data_cache_dir', type=str, default='./data_cache', help='用于存储预分词二进制文件的目录。')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='用于保存模型检查点的目录。')
    parser.add_argument('--resume_from', type=str, default=None, help='从指定的检查点文件路径恢复训练。')

    # --- 模型超参数 ---
    parser.add_argument('--context_length', type=int, default=256, help='模型的最大序列长度（上下文窗口）。')
    parser.add_argument('--num_layers', type=int, default=6, help='Transformer块（层）的数量。')
    parser.add_argument('--d_model', type=int, default=512, help='模型的隐藏层维度。')
    parser.add_argument('--num_heads', type=int, default=8, help='多头注意力机制中的头的数量。')
    parser.add_argument('--d_ff', type=int, default=2048, help='前馈神经网络的中间层维度。')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='旋转位置编码（RoPE）的基数。')

    # --- 优化器超参数 ---
    parser.add_argument('--lr_max', type=float, default=3e-4, help='最大学习率。')
    parser.add_argument('--lr_min', type=float, default=3e-5, help='学习率衰减后的最小学习率。')
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999], help='AdamW优化器的beta参数。')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='AdamW优化器的权重衰减系数。')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的阈值（设置为0则不进行裁剪）。')

    # --- 训练循环控制参数 ---
    parser.add_argument('--batch_size', type=int, default=32, help='训练时的批次大小。')
    parser.add_argument('--max_iters', type=int, default=100000, help='总的训练迭代次数。')
    parser.add_argument('--warmup_iters', type=int, default=2000, help='学习率预热（warmup）的迭代次数。')
    parser.add_argument('--log_interval', type=int, default=10, help='每隔N次迭代打印一次训练日志。')
    parser.add_argument('--eval_interval', type=int, default=500, help='每隔N次迭代在验证集上评估一次。')
    parser.add_argument('--save_interval', type=int, default=1000, help='每隔N次迭代保存一次检查点。')
    parser.add_argument('--seed', type=int, default=42, help='用于可复现性的随机种子。')

    # 解析命令行传入的参数
    args = parser.parse_args()
    # 使用解析到的参数调用主函数
    main(args)