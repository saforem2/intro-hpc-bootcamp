#!/usr/bin/env python3
"""
GPU-scale version of the 01.3 representation-learning experiment.

Runs on a real GPU (Polaris A100). The CPU version in the page shows contrastive
features BARELY beating random init (SSL needs scale). Here we crank the knobs
that matter — bigger batch, more steps, bigger unlabeled pool — to test whether
"contrastive-pretrain + linear-probe beats from-scratch when labels are scarce"
actually holds once you have the compute.

Prints JSON to stdout (last line) so the caller can capture it.
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

t0 = time.time()
torch.manual_seed(0)
np.random.seed(0)

dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"device: {dev}", flush=True)
if dev == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

# ---- data (CIFAR-10: color, harder than FashionMNIST — where SSL should shine) ----
tfm = T.Compose([T.ToTensor()])
train = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=tfm)
test = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)


def take(ds, n, seed):
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n]
    X = torch.stack([ds[i][0] for i in idx])
    y = torch.tensor([ds[i][1] for i in idx])
    return X, y


class Encoder(nn.Module):
    """ResNet-ish small CNN -> 256-d representation."""
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),    # 32->16
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),  # 16->8
            nn.Conv2d(128, 256, 3, 2, 1), nn.BatchNorm2d(256), nn.ReLU(), # 8->4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, dim),
        )

    def forward(self, x):
        return self.net(x)


# strong SimCLR-style augment (GPU tensors)
aug = T.Compose([
    T.RandomResizedCrop(32, scale=(0.3, 1.0), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    T.RandomGrayscale(p=0.2),
])


def nt_xent(z1, z2, temp=0.2):
    z = F.normalize(torch.cat([z1, z2], 0), dim=1)
    n = z1.shape[0]
    sim = z @ z.t() / temp
    sim.fill_diagonal_(-9e15)
    tgt = (torch.arange(2 * n, device=z.device) + n) % (2 * n)
    return F.cross_entropy(sim, tgt)


def pretrain(enc, Xun, steps, bs):
    proj = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128)).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(proj.parameters()), lr=1e-3)
    Xun = Xun.to(dev)
    enc.train()
    curve = []
    for st in range(steps):
        xb = Xun[torch.randint(0, len(Xun), (bs,), device=dev)]
        loss = nt_xent(proj(enc(aug(xb))), proj(enc(aug(xb))))
        opt.zero_grad(); loss.backward(); opt.step()
        if st % (steps // 10) == 0:
            curve.append((st, round(loss.item(), 3)))
    return enc, curve


def probe(enc, Xtr, ytr, Xte, yte, ep=200):
    enc.eval()
    with torch.no_grad():
        Ztr = enc(Xtr.to(dev)); Zte = enc(Xte.to(dev))
    clf = nn.Linear(Ztr.shape[1], 10).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    yt = ytr.to(dev)
    for _ in range(ep):
        opt.zero_grad(); F.cross_entropy(clf(Ztr), yt).backward(); opt.step()
    with torch.no_grad():
        return (clf(Zte).argmax(1) == yte.to(dev)).float().mean().item()


def scratch(Xtr, ytr, Xte, yte, ep=60, bs=64):
    enc = Encoder().to(dev); head = nn.Linear(256, 10).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    Xtr, yt = Xtr.to(dev), ytr.to(dev)
    enc.train()
    for _ in range(ep):
        perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); F.cross_entropy(head(enc(Xtr[b])), yt[b]).backward(); opt.step()
    enc.eval()
    with torch.no_grad():
        return (head(enc(Xte.to(dev))).argmax(1) == yte.to(dev)).float().mean().item()


# ---- run ----
BATCH = 1024 if dev == "cuda" else 256
STEPS = 2000 if dev == "cuda" else 300
POOL = 50000 if dev == "cuda" else 6000

Xun, _ = take(train, POOL, 1)
Xte, yte = take(test, 4000, 2)

enc = Encoder().to(dev)
enc, curve = pretrain(enc, Xun, steps=STEPS, bs=BATCH)
print(f"pretrain done ({time.time()-t0:.0f}s), loss curve: {curve}", flush=True)

rows = []
for n in [100, 250, 500, 1000, 5000]:
    Xtr, ytr = take(train, n, 3)
    a_s = scratch(Xtr, ytr, Xte, yte)
    a_p = probe(enc, Xtr, ytr, Xte, yte)
    rows.append([n, round(a_s, 3), round(a_p, 3), round(a_p - a_s, 3)])
    print(f"n_labels={n:5d}  from-scratch={a_s:.3f}  pretrained+probe={a_p:.3f}  delta={a_p-a_s:+.3f}", flush=True)

result = {
    "device": dev,
    "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
    "dataset": "CIFAR-10",
    "batch": BATCH, "steps": STEPS, "pool": POOL,
    "loss_curve": curve,
    "rows": rows,
    "total_s": round(time.time() - t0, 1),
}
print("RESULT_JSON " + json.dumps(result))
