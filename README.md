# FGSM 对抗攻击：MNIST 复现与最小实现

本仓库包含两部分：

- **最小实现**：[`fgsm_mnist.py`](fgsm_mnist.py)，单文件完成训练与 FGSM 攻击
- **复现**：基于 [1Konny/FGSM](https://github.com/1Konny/FGSM)（原始实现由 WonKwang Lee 提供），修复兼容性问题并做 epsilon 扫描

## 原理

![FGSM](misc/FGSM.PNG)

```
x_adv = x + ε · sign(∇_x J(θ, x, y))
```

## 环境

Python 3.x + PyTorch (CPU)

```bash
pip install torch torchvision pillow matplotlib
```

## 最小实现 fgsm_mnist.py

72 行单文件，包含训练、攻击和评估：

- 分类器：784 → 128 → 10 全连接网络
- 数据：MNIST，`ToTensor()` 后像素范围 [0, 1]
- 训练：Adam lr=1e-3，batch=64，1 epoch
- 攻击：单步 FGSM，ε=0.15，测试集全部 10 000 张
- 固定 `torch.manual_seed(1)`，结果可复现

```bash
python fgsm_mnist.py
```

| Epsilon | Clean Acc | Adversarial Acc |
|---------|-----------|-----------------|
| 0.15 | 94.3% | 1.9% |

## 复现：对 1Konny 代码的改动

- `.data[0]` → `.item()`（PyTorch 0.4+ 不再支持 0 维 tensor 索引）
- Windows 文件名非法字符 `:` → `_`
- 命令显式传 `--dataset MNIST`：默认值 `FMNIST` 会命中 `if 'MNIST' in name` 子串判断，实际加载的仍是 MNIST

### 无目标攻击

| Epsilon | Clean Acc | Adversarial Acc |
|---------|-----------|-----------------|
| 0.03 | 96% | 92% |
| 0.05 | 96% | 90% |
| 0.10 | 96% | 77% |
| 0.20 | 96% | 31% |
| 0.30 | 96% | 7% |

### 有目标攻击（target=3）

| Epsilon | Clean Acc | Accuracy on True Label |
|---------|-----------|------------------------|
| 0.03 | 96% | 96% |
| 0.10 | 96% | 89% |
| 0.20 | 96% | 52% |
| 0.30 | 96% | 23% |

![FGSM comparison](comparison_grid.png)

每行对应一个 epsilon：左列干净样本，中间列对抗样本，右列标记预测是否翻转（绿框未变、红框改变）。由 [`make_comparison_grid.py`](make_comparison_grid.py) 生成。

## 两种实现对比

| 实现 | 分类器 | 输入范围 | ε（像素空间） | Clean Acc | Adversarial Acc |
|------|--------|----------|---------------|-----------|-----------------|
| 1Konny | MLP 784→300→150→10 | [-1, 1] | 0.10（原参数 0.2） | 96% | 31% |
| 本仓库 | MLP 784→128→10 | [0, 1] | 0.15 | 94.3% | 1.9% |

两者都是全连接网络。1Konny 的 ε=0.2 作用在 [-1, 1] 上，等价于像素空间 0.1；最小实现的 ε=0.15 作用在 [0, 1] 上，等价于归一化空间 0.30，扰动约为前者的 1.5 倍。容量（约 28 万 vs 10 万参数）和学习率（2e-4 vs 1e-3）也不同，因此这一差距不能归因于卷积的归纳偏置。

## 运行 1Konny 代码

```bash
# 训练（1 个 epoch）
python main.py --mode train --dataset MNIST --env_name cputorch --cuda False --epoch 1

# 无目标攻击
python main.py --mode generate --dataset MNIST --iteration 1 --epsilon 0.1 --env_name cputorch --cuda False --load_ckpt best_acc.tar

# 有目标攻击（目标类别 3）
python main.py --mode generate --dataset MNIST --iteration 1 --epsilon 0.2 --target 3 --env_name cputorch --cuda False --load_ckpt best_acc.tar
```

## 数据说明

MNIST：60 000 张训练图 / 10 000 张测试图，28×28 灰度，10 类。首次运行自动下载到 `datasets/MNIST/`，不纳入版本控制。

## 目录

- `fgsm_mnist.py`：最小实现
- `main.py` / `solver.py` / `adversary.py` / `models/` / `utils/` / `datasets/datasets.py`：1Konny 代码
- `make_comparison_grid.py` / `comparison_grid.png`：复现实验对比图
- `datasets/`、`checkpoints/`、`output/`：数据、权重与生成图片，不纳入版本控制（后两者保留空目录）
- `misc/`：原仓库示意图

## 参考

1. Goodfellow et al., *Explaining and Harnessing Adversarial Examples*, [arXiv:1412.6572](https://arxiv.org/abs/1412.6572)
2. Kurakin et al., *Adversarial Examples in the Physical World*, [arXiv:1607.02533](https://arxiv.org/abs/1607.02533)
3. [1Konny/FGSM](https://github.com/1Konny/FGSM)
