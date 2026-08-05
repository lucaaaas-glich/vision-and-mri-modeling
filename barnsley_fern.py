import torch
import matplotlib.pyplot as plt


# 1. 选择设备：Mac M2 优先使用 MPS
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# 2. 参数
num_walkers = 20_000   # 同时并行生成多少条轨迹
num_steps = 150        # 每条轨迹迭代多少次
burn_in = 20           # 丢弃前几步，减少初始点影响


# 3. 所有轨迹都从原点开始
x = torch.zeros(num_walkers, device=device)
y = torch.zeros(num_walkers, device=device)

points_x = []
points_y = []


# 4. 迭代生成 Barnsley Fern
for step in range(num_steps):
    # 每条轨迹独立生成一个 0 到 1 的随机数
    r = torch.rand(num_walkers, device=device)

    new_x = torch.empty_like(x)
    new_y = torch.empty_like(y)

    mask1 = r < 0.01
    mask2 = (r >= 0.01) & (r < 0.86)
    mask3 = (r >= 0.86) & (r < 0.93)
    mask4 = r >= 0.93

    # 变换 1：茎底部
    new_x[mask1] = 0.0
    new_y[mask1] = 0.16 * y[mask1]

    # 变换 2：主体叶片
    new_x[mask2] = 0.85 * x[mask2] + 0.04 * y[mask2]
    new_y[mask2] = -0.04 * x[mask2] + 0.85 * y[mask2] + 1.6

    # 变换 3：左侧叶片
    new_x[mask3] = 0.20 * x[mask3] - 0.26 * y[mask3]
    new_y[mask3] = 0.23 * x[mask3] + 0.22 * y[mask3] + 1.6

    # 变换 4：右侧叶片
    new_x[mask4] = -0.15 * x[mask4] + 0.28 * y[mask4]
    new_y[mask4] = 0.26 * x[mask4] + 0.24 * y[mask4] + 0.44

    x, y = new_x, new_y

    if step >= burn_in:
        points_x.append(x.cpu())
        points_y.append(y.cpu())


all_x = torch.cat(points_x).numpy()
all_y = torch.cat(points_y).numpy()


plt.figure(figsize=(6, 10))
plt.scatter(
    all_x,
    all_y,
    c=range(len(all_x)),
    s=0.05,
    cmap="plasma"
)
plt.title("Barnsley Fern")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "images/barnsley_fern.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()