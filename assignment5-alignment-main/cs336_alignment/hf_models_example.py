from jinja2 import optimizer
from torch import device
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch

from tests.conftest import gradient_accumulation_steps

#模型调用
model=AutoModelForCausalLM.from_pretrained(
    "data/models/Qwen2.5-Math-1.5B",torch_dtype=torch.float12,attn_implementation="flash_attention_2"
)
tokenizer=AutoTokenizer.from_pretrained("data/models/Qwen2.5-Math-1.5B")

#数据定义好后进行前向传播
input_ids=train_batch["input_ids"].to(device)
labels=train_batch["labels"].to(device)

logits=model(input_ids).logits
loss=F.cross_entropy(logits,labels)

#训练后的模型保存
model.save_pretrained("models/My_trained")
tokenizer.save_pretrained("models/My_trained")

#这是以往的正常反向传播
for inputs,labels in data_loader:
    logits=model(inputs)
    loss=loss_fn(logits,labels)

    loss.backward()

    optimizer.step()

    optimizer.zero_grad()

#这是梯度积累的反向传播，我们每隔k步调用一次optimizer.step()和zero_grad()，其中k为梯度累计部署，在调用loss反向传播钱，我们将损失函数除以累计步数，使得梯度在累计步数上取平均
gradient_accumulation_steps=4
for idx,(inputs,labels) in enumerate(data_loader):
    logits=model(inputs)
    loss=loss_fn(logits,labels)/gradient_accumulation_steps

    loss.backward()
    if (idx+1)%gradient_accumulation_steps==0:
        optimizer.step()
        optimizer.zero_grad()