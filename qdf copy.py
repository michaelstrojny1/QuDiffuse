
import math, time, random, os
from dataclasses import dataclass
from typing import Tuple, List

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils

# -------------------------
# Repro + CUDA settings
# -------------------------
def set_seed(seed=123):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(123)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
# For PyTorch >= 2.0 on A100 this enables TF32 matmul
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    dataset: str = "MNIST"            # "MNIST" or "CIFAR10"
    class_id: int = 0                 # single class to use
    img_size: int = 28                # MNIST=28; CIFAR gets resized to this
    batch_size: int = 64
    num_workers: int = 0  # Set to 0 for Windows compatibility
    drop_last: bool = True

    # diffusion timesteps
    T: int = 5
    schedule: str = "cosine"          # "cosine" or "linear"
    beta_min: float = 0.01            # used for linear schedule
    beta_max: float = 0.12            # used for linear schedule
    p_star: float = 0.35              # final corruption for cosine schedule; T-invariant

    # model
    hidden_ratio: float = 0.75        # denoising RBM practice ~0.6-0.9
    k_pcd: int = 20                   # PCD steps per update
    lr: float = 2e-3
    wd: float = 1e-4
    grad_clip: float = 5.0
    epochs_per_layer: int = 160

    # sampling/IO
    gibbs_steps_eval: int = 128  # Increased for better quality
    n_samples_grid: int = 64
    out_dir: str = "./runs_cRBM_diffusion"
    save_models: bool = True  # Save trained model layers

cfg = Config()

# -------------------------
# Data (single-class, grayscale, (N,1,H,W) → binarize → (N,D))
# -------------------------
def collate_fn(batch):
    """Collate function for DataLoader (must be at module level for Windows multiprocessing)"""
    xs = torch.stack([b[0] for b in batch], dim=0)  # (B,D)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys
class SingleClassBinaryDataset(Dataset):
    def __init__(self, base, class_id: int):
        # Pre-filter indices to a single label
        self.indices = [i for i, (_, y) in enumerate(base) if int(y) == class_id]
        if len(self.indices) == 0:
            raise RuntimeError(f"No samples for class_id={class_id}.")
        self.base = base
        c, h, w = self.base[0][0].shape
        if c != 1:
            raise AssertionError("Dataset must be grayscale (C=1) after transforms.")
        self.D = h * w

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.base[self.indices[idx]]
        # x: (1,H,W) in [0,1]; Bernoulli sample for binary RBM
        with torch.no_grad():
            x = (torch.rand_like(x) < x).float()
        x = x.view(-1)  # (D,)
        return x, int(y)

def get_dataloaders(cfg: Config):
    if cfg.dataset.upper() == "MNIST":
        tfm = transforms.Compose([
            transforms.ToTensor(),                      # (1,28,28), [0,1]
        ])
        train_base = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        test_base  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    elif cfg.dataset.upper() == "CIFAR10":
        tfm = transforms.Compose([
            transforms.Resize(cfg.img_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),                      # (1,H,W), [0,1]
        ])
        train_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
        test_base  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset")

    train_set = SingleClassBinaryDataset(train_base, cfg.class_id)
    val_set   = SingleClassBinaryDataset(test_base, cfg.class_id)

    # Guard persistent_workers only when workers>0 (avoids multiprocessing teardown issues)
    pw = cfg.num_workers > 0

    train_loader = DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        prefetch_factor=(2 if cfg.num_workers > 0 else None),
        persistent_workers=pw, drop_last=cfg.drop_last, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        prefetch_factor=(2 if cfg.num_workers > 0 else None),
        persistent_workers=pw, drop_last=False, collate_fn=collate_fn
    )
    D = train_set.D
    print(f"Device: {device.type}")
    print(f"Dataset={cfg.dataset} | Class={cfg.class_id} | Train={len(train_set)} | Val={len(val_set)} | D={D}")
    return train_loader, val_loader, D

# -------------------------
# Bit-flip schedule & forward chain
# -------------------------
def cosine_betas(T: int, beta_min: float, beta_max: float, p_star: float = 0.35):
    """
    Cosine schedule that enforces a T-invariant final total corruption p_star.
    beta_min/max are ignored here by design (kept for uniform API).
    Ensures betas in (0, 0.5) for Bernoulli bit-flips.
    """
    t = torch.arange(0, T+1, dtype=torch.float64)       # include 0..T
    alpha = torch.cos(0.5 * math.pi * (t / T))**2       # [0..T], monotone 1→0
    gamma_T = 1.0 - 2.0 * p_star                        # target final product
    gamma_bar = gamma_T * (1 - alpha) + alpha           # gamma_bar[0]=1, gamma_bar[T]=gamma_T
    ratio = (gamma_bar[1:] / gamma_bar[:-1]).clamp(min=1e-6, max=1.0)  # numerical safety
    betas = 0.5 * (1.0 - ratio)
    betas = betas.clamp(min=1e-6, max=0.499)            # strict bit-flip domain
    return betas.float()

def linear_betas(T: int, beta_min: float, beta_max: float):
    betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float32)
    return betas.clamp(min=1e-6, max=0.499)

def build_schedule(cfg: Config):
    if cfg.schedule == "cosine":
        betas = cosine_betas(cfg.T, cfg.beta_min, cfg.beta_max, p_star=cfg.p_star)
    else:
        betas = linear_betas(cfg.T, cfg.beta_min, cfg.beta_max)  # <-- correct order
    if not torch.isfinite(betas).all():
        raise RuntimeError("Non-finite betas in schedule.")
    return betas

@torch.no_grad()
def make_pair_from_x0(x0: torch.Tensor, betas: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x0: (B,D) ∈ {0,1}; betas: (T,); t ∈ [1..T]
    Returns (x_{t-1}, x_t) produced by t-1 and t bit-flip steps.
    """
    B, D = x0.shape
    xtm1 = x0.to(torch.uint8)
    for s in range(1, t):  # 1..t-1
        flip = (torch.rand(B, D, device=x0.device) < betas[s-1]).to(torch.uint8)
        xtm1 = xtm1 ^ flip
    flip_t = (torch.rand(B, D, device=x0.device) < betas[t-1]).to(torch.uint8)
    xt = xtm1 ^ flip_t
    return xtm1.float(), xt.float()

# -------------------------
# cRBM layer
# -------------------------
class CRBM(nn.Module):
    def __init__(self, D: int, H: int, device: torch.device):
        super().__init__()
        self.D, self.H = D, H
        self.device = device

        # Parameters (FP32)
        self.W = nn.Parameter(0.01 * torch.randn(D, H))
        self.a = nn.Parameter(torch.zeros(D))
        self.b = nn.Parameter(torch.zeros(H))

        # Conditional (dynamic biases): a_hat = a + G ⊙ c; b_hat = b + F^T c
        self.G = nn.Parameter(torch.zeros(D))                 # diagonal per-visible gate
        self.F = nn.Parameter(0.01 * torch.randn(D, H))       # dense coupling for hidden

        self.opt = optim.AdamW(self.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

        # PCD chains (uint8 on device)
        self.v_chain: torch.Tensor | None = None
        self.h_chain: torch.Tensor | None = None

    def to(self, *args, **kwargs):
        ret = super().to(*args, **kwargs)
        # keep self.device in sync if caller passes device
        for arg in args:
            if isinstance(arg, torch.device):
                self.device = arg
        if 'device' in kwargs and kwargs['device'] is not None:
            self.device = kwargs['device']
        if self.v_chain is not None:
            self.v_chain = self.v_chain.to(self.device)
        if self.h_chain is not None:
            self.h_chain = self.h_chain.to(self.device)
        return ret

    @torch.no_grad()
    def init_pcd(self, B: int):
        self.v_chain = torch.randint(0, 2, (B, self.D), device=self.device, dtype=torch.uint8)
        self.h_chain = torch.randint(0, 2, (B, self.H), device=self.device, dtype=torch.uint8)

    @torch.no_grad()
    def pcd_k(self, k: int, c: torch.Tensor):
        """
        Persistent Contrastive Divergence steps with clamped condition c.
        Returns (v, h) uint8.
        """
        assert self.v_chain is not None and self.h_chain is not None
        v = self.v_chain
        h = self.h_chain
        for _ in range(k):
            # p(h=1|v,c)
            logits_h = self.b + (c @ self.F) + (v.float() @ self.W)
            logits_h = torch.nan_to_num(logits_h, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            prob_h = torch.sigmoid(logits_h)
            h = (torch.rand_like(prob_h) < prob_h).to(torch.uint8)
            # p(v=1|h,c)
            logits_v = self.a + (self.G * c) + (h.float() @ self.W.t())
            logits_v = torch.nan_to_num(logits_v, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            prob_v = torch.sigmoid(logits_v)
            v = (torch.rand_like(prob_v) < prob_v).to(torch.uint8)
        self.v_chain.copy_(v)
        self.h_chain.copy_(h)
        return v, h

    @torch.no_grad()
    def pll_batch(self, v: torch.Tensor, c: torch.Tensor):
        """
        RBM pseudo-likelihood (clamped c), averaged over batch.
        """
        B, D = v.shape
        i = torch.randint(0, D, (B,), device=v.device)
        mask = F.one_hot(i, num_classes=D).float()

        a_hat = self.a + self.G * c           # (B,D)
        b_hat = self.b + (c @ self.F)         # (B,H)

        v_flip = (v + mask) % 2.0             # flip a bit
        logits_h      = b_hat + v @ self.W
        logits_h_flip = b_hat + v_flip @ self.W

        # Numeric safety before logaddexp
        logits_h = torch.nan_to_num(logits_h, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
        logits_h_flip = torch.nan_to_num(logits_h_flip, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)

        Fv  = -(a_hat * v).sum(dim=1)      - torch.logaddexp(torch.zeros_like(logits_h),      logits_h).sum(dim=1)
        Fvf = -(a_hat * v_flip).sum(dim=1) - torch.logaddexp(torch.zeros_like(logits_h_flip), logits_h_flip).sum(dim=1)

        pll = torch.sigmoid(-(Fvf - Fv)).mean()
        if not torch.isfinite(pll):
            return 0.0
        return float(pll)

    def train_step(self, v_pos: torch.Tensor, c: torch.Tensor, k: int, grad_clip: float):
        """
        v_pos, c: (B,D) floats {0,1}
        Manual MLE gradient via pos/neg stats (no autograd graphs).
        """
        self.opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            a_hat = self.a + self.G * c           # (B,D)
            b_hat = self.b + (c @ self.F)         # (B,H)

            # ----- Positive phase -----
            logits_ph = b_hat + v_pos @ self.W
            logits_ph = torch.nan_to_num(logits_ph, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            ph = torch.sigmoid(logits_ph)                          # (B,H)

            pos_W = v_pos.t() @ ph                                 # (D,H)
            pos_a = v_pos.sum(0)                                   # (D,)
            pos_b = ph.sum(0)                                      # (H,)
            pos_F = c.t() @ ph                                     # (D,H)
            pos_G = (c * v_pos).sum(0)                             # (D,)

            # ----- Negative phase via PCD -----
            v_neg_u8, _ = self.pcd_k(k=k, c=c)
            v_neg = v_neg_u8.float()
            logits_nh = self.b + (c @ self.F) + v_neg @ self.W
            logits_nh = torch.nan_to_num(logits_nh, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            nh = torch.sigmoid(logits_nh)

            neg_W = v_neg.t() @ nh
            neg_a = v_neg.sum(0)
            neg_b = nh.sum(0)
            neg_F = c.t() @ nh
            neg_G = (c * v_neg).sum(0)

        B = max(1, int(v_pos.shape[0]))
        # grads = -(pos - neg)/B (maximize log-likelihood)
        gW = - (pos_W - neg_W).to(torch.float32) / B
        ga = - (pos_a - neg_a).to(torch.float32) / B
        gb = - (pos_b - neg_b).to(torch.float32) / B
        gF = - (pos_F - neg_F).to(torch.float32) / B
        gG = - (pos_G - neg_G).to(torch.float32) / B

        # Replace non-finite gradients with zeros to avoid NaN propagation
        for g in (gW, ga, gb, gF, gG):
            torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0, out=g)

        self.W.grad = gW
        self.a.grad = ga
        self.b.grad = gb
        self.F.grad = gF
        self.G.grad = gG

        # Grad clipping (canonical list form)
        nn.utils.clip_grad_norm_([self.W, self.a, self.b, self.F, self.G], max_norm=grad_clip)

        # Extra safety: if any grad is still non-finite, skip the step
        if not all(torch.isfinite(p.grad).all() for p in (self.W, self.a, self.b, self.F, self.G)):
            self.opt.zero_grad(set_to_none=True)
            return

        self.opt.step()

# -------------------------
# Training per layer
# -------------------------
def train_layer(layer_id: int, crbm: CRBM, train_loader, betas, cfg: Config):
    crbm.train()
    B_fixed = cfg.batch_size
    crbm.init_pcd(B_fixed)

    print(f"\n=== Train layer t={layer_id}/{cfg.T} ===")
    t0 = time.time()
    for ep in range(1, cfg.epochs_per_layer+1):
        pll_meter = []
        iters = 0
        for x0_cpu, _ in train_loader:
            x0 = x0_cpu.to(device, non_blocking=True)  # (B,D) {0,1}
            v_pos, c = make_pair_from_x0(x0, betas, layer_id)

            # Sanity clamps
            v_pos = v_pos.clamp_(0, 1)
            c     = c.clamp_(0, 1)

            pll = crbm.pll_batch(v_pos, c)
            pll_meter.append(pll)

            crbm.train_step(v_pos, c, k=cfg.k_pcd, grad_clip=cfg.grad_clip)
            iters += 1

        avg_pll = sum(pll_meter)/len(pll_meter) if len(pll_meter) > 0 else float('nan')
        if not math.isfinite(avg_pll):
            avg_pll = 0.0
        print(f"t={layer_id} ep={ep}/{cfg.epochs_per_layer} | PLL={avg_pll:.4f} | iters={iters} | {time.time()-t0:.1f}s")
        t0 = time.time()

    # free PCD memory for this layer
    crbm.v_chain = None
    crbm.h_chain = None
    torch.cuda.empty_cache()

# -------------------------
# Reverse sampling (improved with annealing and quality guards)
# -------------------------
@torch.no_grad()
def free_energy(layer: CRBM, v: torch.Tensor, c: torch.Tensor):
    """Compute free energy for quality assessment"""
    a_hat = layer.a + layer.G * c
    b_hat = layer.b + (c @ layer.F)
    logits_h = b_hat + (v @ layer.W)
    logits_h = torch.nan_to_num(logits_h, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
    Fv = -(a_hat * v).sum(dim=1) - torch.logaddexp(torch.zeros_like(logits_h), logits_h).sum(dim=1)
    return Fv

@torch.no_grad()
def extra_gibbs(layer: CRBM, v: torch.Tensor, c: torch.Tensor, steps=256):
    """Extended annealed Gibbs for retry of bad samples"""
    mf_tail = 8
    for s in range(steps):
        # Temperature annealing
        tau = 1.3 - 0.3 * (s / max(1, steps - 1))
        
        logits_h = (layer.b + (c @ layer.F) + (v @ layer.W)) / tau
        logits_h = torch.nan_to_num(logits_h, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
        ph = torch.sigmoid(logits_h)
        
        if s >= steps - mf_tail:
            # Mean-field tail for cleaner results
            logits_v = (layer.a + (layer.G * c) + (ph @ layer.W.t())) / tau
            logits_v = torch.nan_to_num(logits_v, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            pv = torch.sigmoid(logits_v)
            v = (pv > 0.5).float()
        else:
            h = torch.bernoulli(ph)
            logits_v = (layer.a + (layer.G * c) + (h @ layer.W.t())) / tau
            logits_v = torch.nan_to_num(logits_v, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            v = torch.bernoulli(torch.sigmoid(logits_v))
    return v

@torch.no_grad()
def reverse_sample(layers: List[CRBM], cfg: Config, n_samples=64, gibbs_steps=128):
    D = layers[0].D
    x = torch.bernoulli(0.5*torch.ones(n_samples, D, device=device))  # start at x_T ~ Bern(0.5)

    for t in range(cfg.T, 0, -1):
        layer = layers[t-1]
        c = x.clone()
        
        # Conditional warm-start: cheap one-step reconstruction
        logits_h = layer.b + (c @ layer.F) + (c @ layer.W)  # use v=c for initial hidden sample
        logits_h = torch.nan_to_num(logits_h, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
        h = torch.bernoulli(torch.sigmoid(logits_h))
        logits_v = layer.a + (layer.G * c) + (h @ layer.W.t())
        logits_v = torch.nan_to_num(logits_v, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
        v = torch.bernoulli(torch.sigmoid(logits_v))
        
        # Annealed Gibbs with mean-field tail
        mf_tail = 8
        for s in range(gibbs_steps):
            # Temperature annealing: start at 1.3, cool to 1.0
            tau = 1.3 - 0.3 * (s / max(1, gibbs_steps - 1))
            
            logits_h = (layer.b + (c @ layer.F) + (v @ layer.W)) / tau
            logits_h = torch.nan_to_num(logits_h, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
            ph = torch.sigmoid(logits_h)
            
            if s >= gibbs_steps - mf_tail:
                # Mean-field tail: use expectations for cleaner results
                logits_v = (layer.a + (layer.G * c) + (ph @ layer.W.t())) / tau
                logits_v = torch.nan_to_num(logits_v, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
                pv = torch.sigmoid(logits_v)
                v = (pv > 0.5).float()
            else:
                h = torch.bernoulli(ph)
                logits_v = (layer.a + (layer.G * c) + (h @ layer.W.t())) / tau
                logits_v = torch.nan_to_num(logits_v, nan=0.0, posinf=30.0, neginf=-30.0).clamp_(-30, 30)
                v = torch.bernoulli(torch.sigmoid(logits_v))
        
        # Guard against bad tiles (only for final layer t=1)
        if t == 1:
            Fv = free_energy(layer, v, c)
            bad = Fv > (Fv.median() + 2.0 * Fv.std())
            if bad.any():
                # Re-run extended annealed Gibbs for bad samples
                idx = bad.nonzero(as_tuple=True)[0]
                v[idx] = extra_gibbs(layer, v[idx], c[idx], steps=256)
        
        x = v  # becomes x_{t-1}
    return x  # (n,D) {0,1}

def save_grid(x_flat: torch.Tensor, cfg: Config, path: str):
    n = x_flat.shape[0]
    img = x_flat.view(n, 1, cfg.img_size, cfg.img_size).cpu()
    nrow = int(round(math.sqrt(n))) or 1
    # Shuffle tiles for better display (avoid always having same corner pattern)
    grid = utils.make_grid(
        img[torch.randperm(n)], 
        nrow=nrow, padding=2, normalize=False
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)

def save_model(layers: List[CRBM], cfg: Config, out_dir: str):
    """Save trained model layers"""
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save each layer
    for t, layer in enumerate(layers, 1):
        layer_path = os.path.join(model_dir, f"layer_t{t}.pt")
        torch.save({
            'layer_id': t,
            'D': layer.D,
            'H': layer.H,
            'W': layer.W.data,
            'a': layer.a.data,
            'b': layer.b.data,
            'F': layer.F.data,
            'G': layer.G.data,
        }, layer_path)
        print(f"Saved layer t={t} → {layer_path}")
    
    # Save config and metadata
    config_path = os.path.join(model_dir, "config.pt")
    torch.save({
        'config': cfg,
        'T': cfg.T,
        'hidden_ratio': cfg.hidden_ratio,
        'dataset': cfg.dataset,
        'class_id': cfg.class_id,
    }, config_path)
    print(f"Saved config → {config_path}")

def load_model(model_dir: str, device: torch.device):
    """Load trained model layers"""
    config_path = os.path.join(model_dir, "config.pt")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    checkpoint = torch.load(config_path, map_location=device)
    cfg = checkpoint['config']
    
    layers = []
    for t in range(1, cfg.T + 1):
        layer_path = os.path.join(model_dir, f"layer_t{t}.pt")
        if not os.path.exists(layer_path):
            raise FileNotFoundError(f"Layer t={t} not found at {layer_path}")
        
        layer_data = torch.load(layer_path, map_location=device)
        layer = CRBM(D=layer_data['D'], H=layer_data['H'], device=device).to(device)
        
        layer.W.data = layer_data['W']
        layer.a.data = layer_data['a']
        layer.b.data = layer_data['b']
        layer.F.data = layer_data['F']
        layer.G.data = layer_data['G']
        
        layers.append(layer)
        print(f"Loaded layer t={t}")
    
    return layers, cfg

# -------------------------
# Main
# -------------------------
def main(cfg: Config):
    train_loader, _, D = get_dataloaders(cfg)
    H = max(1, int(round(cfg.hidden_ratio * D)))
    print(f"Hidden units H={H}")

    betas = build_schedule(cfg).to(device)
    print(f"Schedule type={cfg.schedule} | T={cfg.T} | first beta={betas[0].item():.4f} | last beta={betas[-1].item():.4f}")

    # build & train layers
    layers: List[CRBM] = []
    for t in range(1, cfg.T+1):
        crbm = CRBM(D=D, H=H, device=device).to(device)
        train_layer(t, crbm, train_loader, betas, cfg)
        layers.append(crbm)

    # Save the trained model
    if cfg.save_models:
        save_model(layers, cfg, cfg.out_dir)
        print("[SUCCESS] Model saved successfully!")

    # Generate samples with improved sampling
    print(f"\nGenerating {cfg.n_samples_grid} samples with {cfg.gibbs_steps_eval} Gibbs steps...")
    x_samples = reverse_sample(layers, cfg, n_samples=cfg.n_samples_grid, gibbs_steps=cfg.gibbs_steps_eval)
    out_path = os.path.join(cfg.out_dir, f"samples_T{cfg.T}_{cfg.dataset}_class{cfg.class_id}.png")
    save_grid(x_samples, cfg, out_path)
    print(f"[SUCCESS] Saved samples -> {out_path}")

if __name__ == "__main__":
    try:
        print("*** Starting MNIST QDF training ***")
        print(f"Using device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        main(cfg)
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        raise