# Part 4 中文复习与演示资料

## 1. 这部分完成了什么

本项目选择了 Medium 路线：

1. Task 1：用卷积 Variational Autoencoder（VAE）重建 OASIS 脑部 MR 图像，并可视化 latent manifold。
2. Task 2：用自建 U-Net 做四类别脑组织分割，使用 categorical/one-hot 输出，并让每个标签的测试 DSC 都超过 0.9。

数据来自 Rangpur：`/home/groups/comp3710/OASIS`。官方已经分成 train、validate 和 test，不能将它们重新混在一起随机切分。

## 2. 数据格式

- MRI 是 256x256、uint8、灰度 PNG。
- segmentation mask 也是 256x256 灰度 PNG。
- 文件通过 `case_227_slice_11.nii.png` 和 `seg_227_slice_11.nii.png` 配对。
- mask 的像素只有 0、85、170、255，代码除以 85 并取整，映射为类别 0、1、2、3。
- 网络训练时缩放到 128x128。MRI 使用 bilinear interpolation；mask 必须使用 nearest-neighbour，否则会制造不存在的类别值。

## 3. VAE 核心原理

普通 Autoencoder 把输入压缩成一个固定向量，VAE 的 encoder 则输出两组参数：

- `mu`：latent distribution 的均值；
- `logvar`：latent distribution 方差的对数。

采样公式是：

```text
std = exp(0.5 * logvar)
epsilon ~ N(0, I)
z = mu + std * epsilon
```

这叫 reparameterisation trick。随机性被移到 `epsilon`，所以 `z` 对 `mu` 和 `logvar` 仍然可微，反向传播能够训练 encoder。

VAE loss：

```text
total loss = reconstruction MSE + beta * KL divergence
```

- Reconstruction loss 要求输出像输入。
- KL divergence 把每张图片的 latent distribution 拉向标准正态分布 `N(0,I)`。
- 没有 KL，模型接近普通 autoencoder，latent space 可能有空洞，随机采样不一定生成合理图像。
- `beta=0.001` 是重建清晰度与 latent 连续性的折中。

### VAE 网络形状（输入 128x128）

```text
[B,1,128,128]
 -> Conv stride 2: [B,32,64,64]
 -> Conv stride 2: [B,64,32,32]
 -> Conv stride 2: [B,128,16,16]
 -> Conv stride 2: [B,256,8,8]
 -> flatten: [B,16384]
 -> mu/logvar: [B,16]
 -> sampled z: [B,16]
 -> decoder: [B,1,128,128]
```

Decoder 使用 transposed convolution 将 latent vector 逐步上采样回图像。最后的 Sigmoid 将像素限制到 `[0,1]`。

### VAE 结果怎么解释

- 训练集：9,664 slices；验证集：1,120 slices。
- 20 epochs，A100 训练 156.0 秒。
- 最佳 validation total loss：0.007456。
- validation reconstruction MSE：0.006039。
- validation KL：1.416143。

重建比输入模糊是正常的，因为 MSE 倾向输出平均、平滑的像素，而且 16 维 latent 是信息瓶颈。manifold 中相邻格子的脑形状连续变化，说明 latent space 不是互相分离的记忆点。

`latent_projection.png` 对 16 维 `mu` 使用 PCA 降到二维。这里的 PCA 只用于画图，不参与 VAE 训练。

## 4. U-Net 核心原理

U-Net 包含两条路径：

- Encoder：卷积提取语义，并通过 max pooling 降低空间尺寸。
- Decoder：逐步上采样，恢复每个像素的位置。

最重要的是 skip connection：把 encoder 同一分辨率的 feature map 与 decoder 拼接。深层 feature 知道“是什么”，浅层 feature 保留“在哪里”和边界细节；分割同时需要两者。

本项目没有调用预建模型。每个 DoubleConv 含两个 3x3 convolution、GroupNorm 和 ReLU。选择 GroupNorm 是因为 MRI 训练 batch 可能较小，它不像 BatchNorm 那样强依赖 batch statistics。

最后一层是 1x1 convolution：

```text
[B,32,H,W] -> [B,4,H,W]
```

这四个通道是每个像素属于四个类别的 logits。训练 target 是 `[B,4,H,W]` one-hot tensor。推理时：

```text
logits -> argmax -> one_hot prediction
```

因此满足 categorical/one-hot output 要求。

## 5. 为什么结合 Cross-Entropy 和 Dice loss

```text
loss = weighted cross-entropy + soft Dice loss
```

- Cross-Entropy 在每个像素上判断类别是否正确。
- Dice loss 直接优化预测区域与真实区域的重叠。
- OASIS 中 background 占约 72.42%，最少的一类只占约 5.46%。只用普通 Cross-Entropy 容易偏向大类别。
- 代码根据训练集像素频率生成 inverse-square-root class weights，最少类别获得更大权重。

## 6. DSC 的含义与计算

Dice Similarity Coefficient：

```text
DSC = 2 * |prediction intersection target| / (|prediction| + |target|)
```

- 完全重合时为 1。
- 完全不重合时接近 0。
- 本项目先在整个数据集上累计每类 intersection 和 denominator，再计算每类 DSC。
- 不能只报告 mean DSC，因为题目要求所有标签都大于 0.9。
- checkpoint 也按照四个 validation DSC 中最低的一个选择，而不是只看平均值。

## 7. U-Net 正式结果

数据量：9,664 train、1,120 validation、544 test。5 epochs 在 A100 上用时 90.6 秒。
虽然训练时输出目录命名为 `part4_unet_pilot`，但 `max_samples` 没有设置，实际使用的是完整官方数据；所有类别达标后，这个 checkpoint 被选为正式模型。

| 类别 | Validation DSC | Test DSC |
| ---: | ---: | ---: |
| 0 | 0.9990 | 0.9990 |
| 1 | 0.9334 | 0.9405 |
| 2 | 0.9453 | 0.9444 |
| 3 | 0.9686 | 0.9685 |

所有 test labels 都超过 0.9。独立 demo inference 得到同样的 DSC，test loss 为 0.097913。

## 8. 老师可能问的问题

### 为什么 mask resize 不能用 bilinear？

Bilinear 会对相邻值做平均。例如 0 和 85 之间会出现 42，这不是合法标签。Nearest-neighbour 只复制已有类别。

### 为什么输出 logits 而不是在 forward 中直接 softmax？

PyTorch Cross-Entropy 内部会用数值更稳定的 log-softmax。训练时直接输入 logits；Dice loss 或推理时才显式 softmax/argmax。

### one-hot 与 class index 有什么区别？

某像素属于类别 2：class index 是 `2`；四类 one-hot 是 `[0,0,1,0]`。one-hot 多一个 class channel，能直接与四通道概率逐类计算 Dice。

### 为什么 background DSC 特别高？

Background 区域大且边界规律，比较容易预测。因此不能只展示整体 accuracy 或平均 DSC，必须检查较小的三个组织类别。

### 为什么 VAE 能生成新图片，而普通 AE 不一定能？

VAE 用 KL regularisation 让 latent distribution 接近连续的标准正态空间，因此可以从 `N(0,I)` 采样再 decode。普通 AE 的 latent vectors 之间可能存在 decoder 没见过的空洞。

### reconstruction 和 generation 的区别？

Reconstruction 的 `z` 来自一张真实输入的 encoder；generation 的 `z` 是从先验分布随机采样或手工设置，再由 decoder 生成，不需要输入图片。

### 这套结果有什么局限？

- 训练使用 2-D slices，没有利用相邻切片的 3-D 信息。
- 为提高速度把 256x256 下采样到 128x128，可能损失细小边界。
- VAE 使用 MSE，生成图会较平滑。
- DSC 很高不代表临床可用；这里只验证课程数据集上的技术性能。

## 9. 演示命令

U-Net 测试集 inference：

```bash
python pattern_recognition_demo2/part4_oasis_unet.py \
  --demo --image-size 128 --base-channels 32 --batch-size 32 --workers 4 \
  --output-dir results/part4_unet
```

VAE 重新生成验证图：

```bash
python pattern_recognition_demo2/part4_oasis_vae.py \
  --demo \
  --data-dir /home/groups/comp3710/OASIS/keras_png_slices_train \
  --val-dir /home/groups/comp3710/OASIS/keras_png_slices_validate \
  --image-size 128 --latent-dim 16 --batch-size 64 --workers 4 \
  --output-dir results/part4_vae
```

演示前需要确认 checkpoint 位于相应 output directory。模型文件没有上传 GitHub，因为体积较大，应保存在本地与 Rangpur。
