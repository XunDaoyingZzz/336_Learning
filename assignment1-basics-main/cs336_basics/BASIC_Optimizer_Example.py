import torch
from collections import defaultdict

class _RequiredParameter(object):
    """单例：用于标注 defaults 中某些参数为必须由 param_group 显式提供的占位符。
    在官方实现中有类似的用法：当某些超参数必须在参数组中指定时会用到这个标记。
    """
    def __repr__(self):
        return "<required parameter>"

# 单例实例，作为 defaults 中“必需但没有默认值”参数的占位
required = _RequiredParameter()

class Optimizer(object):
    """
    PyTorch 优化器基类（教学版，已加入详尽中文注释）
    主要职责：
      - 保存并管理参数（param_groups）
      - 保存优化器状态（state，例如动量缓冲、累积梯度等）
      - 提供 zero_grad / step 等基础接口（step 在子类实现）
    说明：
      - params 可以是可迭代的 Tensor 列表，或是包含多个 param_group 的可迭代对象（每个 group 为 dict）。
      - defaults 是一个 dict，包含该优化器通用的超参数默认值（例如 lr、weight_decay 等）。
    """
    def __init__(self, params, defaults):
        # 官方实现会记录 API 使用情况（用于统计/日志），在教学环境不存在也不影响功能
        try:
            torch._C._log_api_usage_once("python.optimizer")
        except Exception:
            # 非官方/模拟环境可能没有该接口，忽略异常以确保兼容
            pass

        # 保存默认超参数字典（例如 {'lr': 0.01, 'weight_decay': 0}）
        self.defaults = defaults

        # 防止用户误把单个 Tensor 作为 params 传入：要求是可迭代（序列）或 dict 列表
        if isinstance(params, torch.Tensor):
            raise TypeError("params 参数应为可迭代的 Tensors 或 dicts，而不是单个 Tensor")

        # state 用于保存每个参数对应的优化器私有状态（例如 momentum buffer、exp avg 等）
        # 结构：defaultdict(dict)，键通常为参数对象（Tensor），值为 dict（存放该参数的状态条目）
        # 例：self.state[p]['momentum_buffer'] = some_tensor
        self.state = defaultdict(dict)

        # param_groups 列表，每一项是 dict（至少需包含 'params' 键）
        # 每个 param_group 可以拥有自己的超参数（例如针对不同层使用不同 lr）
        self.param_groups = []

        # 将 params 转换为 list 以便统一处理
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer 得到的是空的参数列表")

        # 如果传入的第一个元素不是 dict，说明用户直接传入了参数序列（比如 model.parameters()），
        # 将该序列封装为一个 param_group：{'params': [p1, p2, ...]}
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        # 逐个添加 param_group（add_param_group 会执行完整校验与补默认值）
        for param_group in param_groups:
            self.add_param_group(param_group)

    def __getstate__(self):
        """
        用于序列化（pickle）优化器时，指定要保存的状态子集。
        通常保存 defaults, state, param_groups 三个字段即可恢复优化器行为。
        """
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        """
        反序列化时恢复到对象字典中。
        注意：反序列化后，某些内部对象（例如 Tensor 对象）需要与模型的参数对象一一对应，
        否则 state 中的键可能无法正确匹配到当前模型参数。
        """
        self.__dict__.update(state)

    def add_param_group(self, param_group):
        """
        向优化器动态添加一个参数组（常用于 fine-tuning，添加新层时给新参数组设置特殊 lr）。
        param_group: dict，必须包含键 'params'（可为单个 Tensor、可迭代 Tensor、list 等）。
        该函数会做如下校验与处理：
          - 确保 param_group 是 dict
          - 将单个 Tensor 转为列表，将可迭代转换为列表
          - 禁止使用 set（无序）作为 params
          - 校验每个 param 为 Tensor 且为 leaf（叶子节点）
          - 将 defaults 中的默认超参数补齐到 param_group 中（若 defaults 中某项为 required，则必须显式提供）
          - 检查参数是否重复分配到多个 param_group
        """
        # 必须是 dict 类型
        if not isinstance(param_group, dict):
            raise TypeError("param_group 必须是 dict")

        # 必须包含 'params' 键
        params = param_group.get('params')
        if params is None:
            raise KeyError("param_group 必须包含键 'params'")

        # 允许用户传入单个 Tensor：转为列表
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        # 不允许 set（无序，会导致不可重复结果）
        elif isinstance(params, set):
            raise TypeError('optimizer 参数需要是有序集合；set 的顺序在不同运行间可能变化，请使用 list')
        else:
            # 将任何可迭代对象转换为 list，避免重复迭代产生副作用
            param_group['params'] = list(params)

        # 对每个 param 做类型与 leaf 校验
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer 只能优化 Tensor，但发现了其他类型: {}".format(type(param)))
            # 只有叶子节点可以被优化（模型参数默认是 leaf），非叶子一般由计算图中间结果产生
            if not param.is_leaf:
                raise ValueError("不能优化非叶子 Tensor（请确保传入的是模型参数）")

        # 将 defaults 中的默认超参数填入 param_group（如果 param_group 没有该超参）
        for name, default in self.defaults.items():
            # 如果 defaults 中某项被标记为 required，则要求 param_group 必须显式提供该值
            if default is required and name not in param_group:
                raise ValueError("param_group 未指定必需的优化参数: {}".format(name))
            else:
                # 如果 param_group 未给出该超参数，则使用 defaults 的值
                param_group.setdefault(name, default)

        # 检查该 param_group 中的参数是否与已有 param_groups 重复
        # 不能让同一个参数在多个组中出现（语义冲突：两个组可能使用不同 lr）
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("某些参数在多个参数组中出现（不允许）")

        # 校验全部通过，添加到 self.param_groups
        self.param_groups.append(param_group)

    def zero_grad(self, set_to_none: bool = False):
        """
        清除（重置）网络中所有优化参数的梯度。
        参数:
          - set_to_none (bool): 如果为 True，将 p.grad 设为 None；否则用原地清零 p.grad.zero_()。
        设计考量：
          - 将 grad 设为 None 可以帮助减少一部分内存且显示地表示“没有梯度”状态；
            有些 PyTorch 内部/扩展对 grad 为 None 有特殊优化。
          - detach_() 用于切断梯度与计算图的连接，避免梯度张量保留不必要的计算图，导致内存泄漏。
        常见用法：
          - 在每次 optimizer.step() 之后或下一轮梯度累积开始前调用 optimizer.zero_grad()
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if set_to_none:
                    # 直接置 None，下一次反向传播会重新分配 grad
                    p.grad = None
                else:
                    # detach_() 把 grad 从计算图分离，zero_() 原地清 0
                    p.grad.detach_()
                    p.grad.zero_()

    def step(self, closure=None):
        """
        执行单次参数更新（核心方法，子类必须实现具体优化算法）。
        参数:
          - closure (callable, optional): 一些优化算法（如 LBFGS）需要多次评估 loss 并回传 loss 值，
            这时可传入一个闭包，closure() 在内部会重新计算 forward+loss+backward，并返回 loss。
        返回值：
          - 一般不返回值；当使用 closure 时，可能返回 closure() 的结果（例如 loss）。
        注意：
          - 这是一个抽象方法，这个基类不实现具体的更新逻辑。
        """
        raise NotImplementedError
