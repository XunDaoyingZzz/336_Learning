import os
import torch
import torch.distributed as dist

"""
编辑配置->环境变量，主机副机按以下两个设置
主机：MASTER_ADDR=192.168.100.1;MASTER_PORT=29500;WORLD_SIZE=2;RANK=0;NCCL_SOCKET_IFNAME=eth0;NCCL_DEBUG=INFO
副机(地址是192.168.100.2)：MASTER_ADDR=192.168.100.1;MASTER_PORT=29500;WORLD_SIZE=2;RANK=1;NCCL_SOCKET_IFNAME=eth1;NCCL_DEBUG=INFO

这里的东西的设定下面会详细说

配置过程：
两台机器都用wsl 在终端都输入ip addr，查看信息，忽略 lo、wlan、docker、br-这些，直接看eth0 eth1这种信息 确保状态是<BROADCAST,MULTICAST,UP,LOWER_UP>然后看IP，如果IP已经配置了，我们能看到192.168.x.x
否则没有配置。我们进行配置

先确保wsl版本足够新，能够兼容：在powershell中运行 wsl --update
然后配置.wslconfig:在windows的资源管理器上方地址栏输入 %UserProfile% 找.wslconfig，如果没有，我们新建记事本写一个.wslconfig，内容如下
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true

另存在 %UserProfile% 里面，注意文件名是.wslconfig，然后格式是所有文件
配置好之后在powershell中:wsl --shutdown
然后重启wsl:wsl
两台机器输入 ip addr，这个时候就能看见eth中有192.168.x.x，可以在副机上的bash：ping 192.168.100.1检查链接

然后两个机器写同一个脚本，搞好一开始的编辑配置，注意添加脚本地址，先后运行即可 
"""


def run():
    # 从环境变量获取配置
    # 这些变量我们在 PyCharm 的设置里填进去，不需要改代码
    rank = int(os.environ["RANK"]) #主机是0，副机是1
    local_gpu_id = 0  # 单机单卡，默认用设备 0

    print(f"--> [进程 {rank}] 正在启动，准备连接 Master...")

    # 初始化分布式后端
    try:
        dist.init_process_group(backend="nccl", init_method="env://")  #init_method指定 env://会读取我们写在编辑配置里的MASTER_ADDR和MASTER_PORT，然后主机启动一个服务器监听29500端口，等待副机进来连
        print(f"--> [进程 {rank}] NCCL 初始化成功！握手完成。")
    except Exception as e:
        print(f"!! [进程 {rank}] 初始化失败。请检查防火墙或网络设置。\n错误信息: {e}")
        return

    # 设置 GPU
    torch.cuda.set_device(local_gpu_id)
    device = torch.device(f"cuda:{local_gpu_id}")

    # 准备数据
    # Rank 0 拿着数字 10，Rank 1 拿着数字 20
    tensor = torch.tensor([10.0 if rank == 0 else 20.0], device=device)
    print(f"    [进程 {rank}] 归约前的数据: {tensor.item()}")

    # 执行 All-Reduce (求和)
    # 这一步会互相等待，直到两边都运行到这里
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # 验证结果 (10 + 20 应该等于 30)
    result = tensor.item()
    print(f"    [进程 {rank}] 归约后的数据: {result}")

    if result == 30.0:
        print(f"SUCCESS: [进程 {rank}] 通信成功！双机计算正确。")
    else:
        print(f"FAIL: [进程 {rank}] 结果不对。")

    dist.destroy_process_group()


if __name__ == "__main__":
    run()
