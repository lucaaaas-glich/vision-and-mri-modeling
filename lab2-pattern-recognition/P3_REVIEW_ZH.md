# COMP3710 Lab 2 Part 3 复习与答辩资料

> 目标：不仅能运行代码，还能解释数据、网络、训练、评估和 A100 实验结果。

## 1. Part 3 评分要求

Part 3 共 **5 分**。

### 3.1 LFW CNN（1 分）

- 使用与 Part 2 相同的 LFW 人脸数据集；
- 两个 `3×3` convolution layers；
- 每个 convolution layer 有 32 个 filters；
- 使用 dense/fully-connected layers 完成分类；
- 可使用 Adam 和 sparse categorical cross-entropy；
- 目标是超过 Part 2 的 PCA + Random Forest。

### 3.2 DAWNBench（4 分）

- 自己构建 CIFAR-10 ResNet-18，不使用未经批准的预建模型；
- 超过 90% accuracy，并在集群上快速训练（1 分）；
- 在 Rangpur 演示 inference 和一个 training epoch（1 分）；
- 使用 mixed precision，在约 360 秒或更快达到 94%（2 分）。

## 2. 本次实验结果

| 实验 | 最佳测试准确率 | 时间/说明 |
| --- | ---: | --- |
| P2：PCA + Random Forest | 60.87% | 150 PCs，保留 94.72% variance |
| P3.1：LFW CNN | 87.27% | 20 epochs，Apple M2 MPS |
| P3.2：CIFAR-10 ResNet-18 | **94.34%** | 55 epochs，A100 mixed precision |

P3.2 的纯训练时间是 **341.5 秒**，完整 Slurm job 是 **348 秒**，都低于
360 秒。现场演示测试得到：

```text
Inference accuracy: 94.340%
One demo training epoch: 7.9 seconds
```

## 3. P2 与 P3 的根本区别

### P2

```text
图片 → 立即 flatten → PCA → 150维 face space → Random Forest
```

- PCA 是无监督、线性降维；
- PCA 最大化 variance，不直接优化分类准确率；
- PCA 和 Random Forest 分开训练；
- 原图一开始就被 flatten，空间关系被打散。

### P3

```text
二维图片 → Conv/ReLU/Pool → Flatten → Dense → logits
```

- CNN 保留二维空间结构；
- 可学习非线性、层次化特征；
- feature extractor 和 classifier 端到端联合训练；
- loss 会直接指导 filters 学习有助于分类的特征。

### 标准回答

> PCA finds directions with high overall variance, but those directions are not
> necessarily the most discriminative for identity. The CNN preserves spatial
> structure and learns task-specific features end-to-end from the classification
> loss, so it substantially outperforms PCA plus Random Forest.

## 4. P3.1 数据准备

### 为什么用 `.images` 而不是 `.data`？

```python
X = lfw_people.images
```

```text
.images: [1288, 50, 37]  保留 height 和 width
.data:   [1288, 1850]    已经 flatten
```

CNN 需要相邻像素和局部结构，因此使用 `.images`。

### 数据划分

```python
train_test_split(X, y, test_size=0.25, random_state=42)
```

```text
训练集：966
测试集：322
```

P2 和 P3 使用相同划分，accuracy 比较才公平。

### NCHW

PyTorch `Conv2d` 要求：

```text
[N, C, H, W] = [batch, channel, height, width]
```

灰度图加入 channel dimension：

```python
X_train = X_train[:, np.newaxis, :, :]
```

```text
[966, 50, 37] → [966, 1, 50, 37]
```

RGB CIFAR-10 的输入则是：

```text
[batch, 3, 32, 32]
```

### dtype

- images 使用 `float32`；
- integer class labels 使用 `int64/torch.long`；
- `CrossEntropyLoss` 的标签不是 one-hot，而是 `0...C-1` 的类别索引。

### DataLoader

```text
train: batch_size=32, shuffle=True
test:  batch_size=32, shuffle=False
```

训练集 shuffle 可以避免固定类别顺序影响 mini-batch 更新。测试集不需要 shuffle。

## 5. P3.1 FaceCNN 尺寸

```text
输入                         [B,  1, 50, 37]
Conv2d(1,32,3,padding=1)    [B, 32, 50, 37]
ReLU                         [B, 32, 50, 37]
MaxPool2d(2)                 [B, 32, 25, 18]
Conv2d(32,32,3,padding=1)   [B, 32, 25, 18]
ReLU                         [B, 32, 25, 18]
MaxPool2d(2)                 [B, 32, 12,  9]
Flatten                      [B, 3456]
Linear(3456,128)             [B, 128]
ReLU + Dropout(0.3)          [B, 128]
Linear(128,7)                [B, 7]
```

其中：

```text
3456 = 32 × 12 × 9
```

### 卷积输出公式

```text
output = floor((input + 2P - K) / S) + 1
```

对 `K=3, P=1, S=1`：

```text
50 → 50
37 → 37
```

对 `MaxPool2d(2)`：

```text
50 → 25 → 12
37 → 18 → 9
```

奇数除以 2 时默认向下取整。

## 6. P3.1 参数量

### Conv1

```text
32 × (1 × 3 × 3) + 32 biases = 320
```

### Conv2

第二层的每个 filter 必须观察全部 32 个输入 channels：

```text
32 × (32 × 3 × 3) + 32 biases = 9,248
```

### Dense layers

```text
Linear 3456→128: 3456×128 + 128 = 442,496
Linear 128→7:    128×7 + 7       = 903
```

### 总参数量

```text
320 + 9,248 + 442,496 + 903 = 452,967
```

ReLU、pooling、flatten 和 dropout 没有可学习参数。

## 7. 各层作用

### Convolution

- 局部连接；
- 权重共享；
- 第一层常学习边缘、方向和纹理；
- 后续层组合低级特征，学习眼睛、鼻子和脸部结构。

### ReLU

```text
ReLU(x) = max(0, x)
```

加入 non-linearity。没有 activation 时，多层线性变换仍等价于一个线性变换。

### MaxPool

- 每个局部区域保留最大 response；
- 降低空间尺寸和计算量；
- channel 数量不变；
- 提供一定的位置不敏感性。

### Flatten

将卷积输出转换为 dense layer 可接收的向量，但保留 batch dimension：

```text
[B,32,12,9] → [B,3456]
```

### Dropout

`Dropout(0.3)` 在训练时随机关闭 30% activation，缓解 overfitting。测试时
`model.eval()` 会关闭 dropout。

### Logits

最后输出 `[B,7]` raw logits，不在模型末尾手动加 Softmax，因为
`CrossEntropyLoss` 内部会以数值稳定的方式处理它。

## 8. 训练循环

必须记住以下顺序：

```python
optimizer.zero_grad()
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

对应含义：

```text
zero_grad       清除上一 batch 累积的梯度
forward         由输入得到 logits
loss            测量预测与真实标签的差异
backward        用链式法则计算所有参数的 gradient
optimizer.step  Adam/SGD 根据 gradient 更新参数
```

如果不执行 `zero_grad()`，PyTorch 默认会累加梯度。

### Train 与 evaluation

训练：

```python
model.train()
torch.set_grad_enabled(True)
```

测试：

```python
model.eval()
torch.no_grad()  # 或 inference_mode
```

`model.eval()` 改变 Dropout/BatchNorm 行为，但不会自动关闭 gradient；关闭 gradient
需要 `no_grad()` 或 `inference_mode()`。

## 9. P3.1 结果分析

```text
Train accuracy: 92.13%
Test accuracy:  87.27%
```

约 5 个百分点的 generalization gap 表示轻微 overfitting。后期 training loss
继续下降，而 test loss 在约 0.47–0.51 波动，也是 overfitting 的信号。

早期 test accuracy 可能高于 train accuracy，因为训练时 Dropout 开启，而测试时关闭。

### Class imbalance

测试集 support：

```text
Ariel Sharon:     13
George W Bush:   146
```

Bush 样本更多，因此模型通常在该类表现最好。不能只看 overall accuracy，还应看：

- precision；
- recall；
- F1-score；
- macro average；
- weighted average。

`macro F1` 给所有类别相同权重，较能暴露小类别表现差的问题；`weighted F1` 按
support 加权，容易被大类别主导。

## 10. ResNet-18 核心

### 为什么需要 residual connection？

普通深层网络直接学习目标映射 `H(x)`。ResNet 改为学习 residual：

```text
F(x) = H(x) - x
H(x) = F(x) + x
```

代码核心：

```python
identity = self.shortcut(x)
out = self.bn2(self.conv2(out))
out = self.relu(out + identity)
```

shortcut 为梯度提供更直接的传播路径，使深层网络更容易优化，并缓解 degradation
problem。

### ResNet-18 为什么是 18 层？

```text
stem convolution:       1
8 BasicBlocks × 2 conv: 16
final fully connected:  1
总计:                   18
```

四个 stage 的 block 数量：

```text
[2, 2, 2, 2]
```

channels：

```text
64 → 128 → 256 → 512
```

后三个 stage 首个 block 使用 `stride=2` 做 downsampling。

### 为什么 CIFAR stem 是 3×3？

ImageNet ResNet 常用 `7×7, stride=2` 和早期 max-pool，但 CIFAR-10 只有
`32×32`。过早大幅 downsample 会丢失太多信息，因此本实现使用：

```text
3×3, stride=1, padding=1
```

### shortcut 什么时候需要投影？

当 stride 或 channels 改变时，`F(x)` 与 `x` 形状不同，不能直接相加。使用：

```text
1×1 convolution + BatchNorm
```

将 identity 投影到相同 shape。

## 11. CIFAR-10 数据与增强

输入：

```text
[B,3,32,32]
```

训练 transforms：

```text
RandomCrop(32, padding=4)
RandomHorizontalFlip()
ToTensor()
Normalize(mean, std)
```

随机裁剪和翻转只用于训练，用来人为增加合理变化、降低 overfitting。测试集不能使用
随机增强，否则每次 evaluation 的输入都会变化。

Normalization 按 RGB channel 减均值、除标准差，使输入尺度更稳定。

## 12. CIFAR-10 优化方法

```text
Optimizer:       SGD
Momentum:        0.9
Weight decay:    5e-4
Nesterov:        True
Label smoothing: 0.1
LR schedule:     CosineAnnealing
Batch size:      256
Epochs:          55
```

### SGD momentum

累积过去的更新方向，减少震荡并加速沿稳定方向前进。

### Weight decay

惩罚过大的 weights，是一种 regularization，可改善泛化。

### Label smoothing

避免模型对 one-hot 标签过度自信，可改善泛化与 calibration。因此即使训练 accuracy
接近 100%，loss 也不会接近完全的 0。

### Cosine schedule

学习率按 cosine 曲线逐渐下降：前期进行较大更新，后期用小学习率精细收敛。

## 13. Mixed precision

CUDA 上使用：

```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(images)
    loss = criterion(logits, labels)
```

优势：

- Tensor Cores 对 FP16 运算速度快；
- 降低显存占用；
- 提高吞吐量。

`GradScaler` 会放大 loss，降低 FP16 中很小的 gradient 下溢为 0 的风险：

```python
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

模型并不是所有操作都强制变成 FP16；autocast 会为不同操作选择合适精度。

## 14. 为什么 demo train accuracy 低于 inference accuracy？

```text
Inference accuracy: 94.34%
Demo train accuracy: 87.688%
```

两者不能直接比较：

- inference 使用干净 test set；
- training loader 使用随机裁剪和翻转；
- 模型处于 training mode；
- demo 创建了新的 optimizer，并以初始学习率执行更新；
- demo 的目标是证明能运行一个训练 epoch，不是重新报告最终模型准确率。

## 15. 高频答辩问题

### Q1：为什么 CNN 比 PCA + Random Forest 好？

CNN 保留空间结构，并利用分类 loss 端到端学习非线性、任务相关特征；PCA 只寻找最大
variance 的线性方向，与分类器分开训练。

### Q2：32 filters 是什么意思？

每个 filter 产生一张 feature map，所以该卷积层输出 32 channels。

### Q3：第二层 filter 的尺寸是什么？

输入有 32 channels，所以每个 filter 是 `32×3×3`，整体 weight tensor 是
`[32,32,3,3]`。

### Q4：为什么不在最后加 Softmax？

因为 `CrossEntropyLoss` 接收 raw logits，并在内部执行数值稳定的 log-softmax 相关
计算。

### Q5：`backward()` 会更新参数吗？

不会。它只计算 gradient；真正更新参数的是 `optimizer.step()`。

### Q6：`eval()` 会关闭 gradient 吗？

不会。`eval()` 改变 Dropout/BatchNorm 行为；还需要 `no_grad()` 或
`inference_mode()` 关闭梯度记录。

### Q7：为什么需要 BatchNorm？

它使用 batch statistics 规范化中间 activations，通常能稳定和加速训练。训练时更新
running statistics，测试时使用保存的 running statistics。

### Q8：为什么 residual block 相加前有时需要 1×1 convolution？

因为 stride 或 channel 数量改变后，identity 与 residual branch shape 不同；1×1
projection 让它们形状一致。

### Q9：为什么 94.34% 不是最后 epoch 的 94.20%？

程序每当 test accuracy 创新高就保存 checkpoint。最佳结果出现在 epoch 53，因此最终
报告 best accuracy，而不是强制使用 epoch 55。

### Q10：如何证明没有使用预建 ResNet？

代码自己定义了 `BasicBlock`、`_make_layer()`、四个 stages 和 shortcut projection，
没有调用 `torchvision.models.resnet18()`，也没有加载预训练 weights。

## 16. 容易犯的错误

- 把 P3 输入写成 `.data`，导致空间结构丢失；
- 忘记 channel dimension；
- 把 PyTorch NCHW 写成 NHWC；
- 第二层 `in_channels` 错写成 1；
- Linear 的 `in_features` 与 flatten size 不匹配；
- 在 CrossEntropyLoss 前手动加 Softmax；
- 标签使用 float 而不是 long；
- 忘记 `zero_grad()`，导致梯度累积；
- 测试时忘记 `model.eval()`；
- 认为 `eval()` 会自动关闭 gradient；
- 在 login node 上直接训练；
- Slurm 忘记使用 `-A comp3710`；
- 把训练 accuracy 当成 test accuracy；
- 把最后 epoch 当成最佳 checkpoint；
- 只报告 accuracy，不分析 class imbalance 和 F1。

## 17. 30 秒答辩版本

> In Part 3.1, I replaced the PCA and Random Forest pipeline with an end-to-end
> two-layer CNN. I kept the LFW images in NCHW form, used two 3-by-3 convolution
> layers with 32 filters, and classified the extracted features with dense
> layers. It achieved 87.27% test accuracy, compared with 60.87% for PCA plus
> Random Forest. In Part 3.2, I implemented ResNet-18 directly for CIFAR-10,
> using residual connections, data augmentation, SGD with cosine decay, and CUDA
> mixed precision. On a Rangpur A100 it reached 94.34% accuracy in 341.5 seconds.
> I also verified live inference at 94.34% and one additional training epoch in
> 7.9 seconds.

## 18. 自测题

1. 为什么 `[966,50,37]` 要变成 `[966,1,50,37]`？
2. `Conv2d(1,32,3,padding=1)` 的输出 shape 和参数量是多少？
3. 为什么第二个 convolution 有 9,248 个参数？
4. 两次 pooling 后为什么是 `12×9`？
5. 为什么 flatten size 是 3,456？
6. 为什么 CrossEntropyLoss 前不加 Softmax？
7. `model.eval()` 与 `torch.no_grad()` 有什么不同？
8. `loss.backward()` 与 `optimizer.step()` 分别做什么？
9. 为什么 residual connection 有助于训练深层网络？
10. 什么时候 shortcut 需要 1×1 convolution？
11. 为什么 CIFAR-10 不沿用 ImageNet 的 7×7 stem？
12. mixed precision 为什么更快？GradScaler 解决什么问题？
13. 为什么 94.34% 是 best accuracy，而最后 epoch 是 94.20%？
14. 为什么 macro F1 会明显低于 weighted F1？
15. 如何证明实验满足 DAWNBench 的准确率和时间要求？

