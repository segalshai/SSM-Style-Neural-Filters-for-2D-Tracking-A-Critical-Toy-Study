import matplotlib
matplotlib.use("Agg")  # MUST come before pyplot

import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


# ============================================================
# Marketing-friendly demo (FIXED for NaNs):
# Sparse measurements + long dropout inside a turn
#
# Fix summary:
# - Bootstraps velocity from first two REAL measurements (m=1) to avoid v=0 blow-up
# - Removes unsafe "min_speed normalization" when velocity is not initialized
# - Caps the normalization scale factor to prevent huge multipliers
# - Slightly smaller learning rate + grad clip already in place
# ============================================================


class SimpleMambaLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.dt_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def reset_stream(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        self.h = torch.zeros(batch_size, self.d_model, device=device, dtype=dtype)

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        x_gate = self.in_proj(x_t)
        x_part, gate = x_gate.chunk(2, dim=-1)
        dt = torch.sigmoid(self.dt_proj(x_part))
        self.h = (1.0 - dt) * self.h + dt * x_part
        y_t = self.out_proj(self.h * torch.sigmoid(gate))
        return y_t


class PredictorNet(nn.Module):
    def __init__(self, in_dim=7, d_model=64, omega_cap=2.2):
        super().__init__()
        self.omega_cap = omega_cap
        self.embed = nn.Linear(in_dim, d_model)
        self.m1 = SimpleMambaLayer(d_model)
        self.m2 = SimpleMambaLayer(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head_omega = nn.Linear(d_model, 1)

    def reset_stream(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        self.m1.reset_stream(batch_size=batch_size, device=device, dtype=dtype)
        self.m2.reset_stream(batch_size=batch_size, device=device, dtype=dtype)

    def step(self, u_t: torch.Tensor) -> torch.Tensor:
        x = self.embed(u_t)
        y1 = self.m1.step(x)
        h1 = x + y1
        y2 = self.m2.step(h1)
        h2 = self.norm(h1 + y2)
        omega = self.head_omega(h2)
        omega = self.omega_cap * torch.tanh(omega)
        return omega


class UpdateNet(nn.Module):
    def __init__(self, in_dim=7, d_model=64):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.m1 = SimpleMambaLayer(d_model)
        self.m2 = SimpleMambaLayer(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head_dx = nn.Linear(d_model, 4)

    def reset_stream(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        self.m1.reset_stream(batch_size=batch_size, device=device, dtype=dtype)
        self.m2.reset_stream(batch_size=batch_size, device=device, dtype=dtype)

    def step(self, u_t: torch.Tensor) -> torch.Tensor:
        x = self.embed(u_t)
        y1 = self.m1.step(x)
        h1 = x + y1
        y2 = self.m2.step(h1)
        h2 = self.norm(h1 + y2)
        dx = self.head_dx(h2)
        return dx


class LearnedSparseFilter(nn.Module):
    def __init__(self, dt=0.1, d_model=64, omega_cap=2.2):
        super().__init__()
        self.dt = dt
        self.pred_net = PredictorNet(in_dim=7, d_model=d_model, omega_cap=omega_cap)
        self.upd_net = UpdateNet(in_dim=7, d_model=d_model)

        # Safety / stabilization knobs
        self.max_vel = 6.0        # clamp velocity magnitude when bootstrapping
        self.max_scale = 3.0      # clamp normalization scale
        self.eps = 1e-6

    def reset_stream(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        self.x = torch.zeros(batch_size, 4, device=device, dtype=dtype)
        self.inited = False

        # Track real measurements for velocity bootstrap
        self.have_prev_meas = False
        self.prev_meas = torch.zeros(batch_size, 2, device=device, dtype=dtype)
        self.vel_inited = False

        self.pred_net.reset_stream(batch_size=batch_size, device=device, dtype=dtype)
        self.upd_net.reset_stream(batch_size=batch_size, device=device, dtype=dtype)

    def init_from_measurement(self, z0: torch.Tensor):
        self.x[:, 0:2] = z0
        self.x[:, 2:4] = 0.0
        self.inited = True

        # Use the first real measurement as previous reference
        self.prev_meas = z0.clone()
        self.have_prev_meas = True
        self.vel_inited = False

    @staticmethod
    def rotate(vx, vy, ang):
        c = torch.cos(ang)
        s = torch.sin(ang)
        vx2 = c * vx - s * vy
        vy2 = s * vx + c * vy
        return vx2, vy2

    def _bootstrap_velocity_if_possible(self, z_real: torch.Tensor, m: torch.Tensor):
        # Only bootstrap on real measurements
        if float(m[0, 0].item()) < 0.5:
            return

        if not self.have_prev_meas:
            self.prev_meas = z_real.clone()
            self.have_prev_meas = True
            return

        if not self.vel_inited:
            dv = (z_real - self.prev_meas) / self.dt  # [B,2]
            # Clamp magnitude to avoid explosions
            mag = torch.sqrt((dv * dv).sum(dim=-1, keepdim=True) + self.eps)
            scale = torch.clamp(self.max_vel / mag, max=1.0)
            dv = dv * scale
            self.x[:, 2:4] = dv
            self.vel_inited = True

        # Always update prev_meas when measurement is real
        self.prev_meas = z_real.clone()

    def step(self, z_in: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # Initialize from first available measurement
        if not self.inited:
            if float(m[0, 0].item()) > 0.5:
                self.init_from_measurement(z_in)
            else:
                self.inited = True

        # Bootstrap velocity from real measurements (m=1)
        self._bootstrap_velocity_if_possible(z_in, m)

        # Predict omega from history + inputs
        u_pred = torch.cat([self.x, z_in, m], dim=-1)  # [B,7]
        omega = self.pred_net.step(u_pred)             # [B,1]
        ang = omega[:, 0] * self.dt

        px, py, vx, vy = self.x[:, 0], self.x[:, 1], self.x[:, 2], self.x[:, 3]

        # Rotate velocity direction
        vx_r, vy_r = self.rotate(vx, vy, ang)

        # Avoid unstable normalization: only normalize if velocity already initialized
        if self.vel_inited:
            vnr = torch.sqrt(vx_r * vx_r + vy_r * vy_r + self.eps)
            v0 = torch.sqrt(vx * vx + vy * vy + self.eps)
            scale = torch.clamp(v0 / vnr, max=self.max_scale)
            vx_new = vx_r * scale
            vy_new = vy_r * scale
        else:
            # Before velocity init, do not force magnitude changes
            vx_new = vx_r
            vy_new = vy_r

        # Position propagation
        px_new = px + vx_new * self.dt
        py_new = py + vy_new * self.dt
        x_pred = torch.stack([px_new, py_new, vx_new, vy_new], dim=-1)

        # Learned update only when m=1
        innov = z_in - x_pred[:, 0:2]
        u_upd = torch.cat([x_pred, innov, m], dim=-1)
        dx = self.upd_net.step(u_upd)

        # Optional small clamp on dx to avoid spikes
        dx = torch.clamp(dx, -5.0, 5.0)

        self.x = x_pred + dx * m
        return self.x


class KalmanCV:
    def __init__(self, dt=0.1, q=1e-4, r=0.04):
        self.dt = dt
        self.q = q
        self.r = r

        self.x = torch.zeros(4)
        self.P = torch.eye(4) * 10.0

        self.F = torch.tensor(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], dtype=torch.float32
        )
        self.H = torch.tensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], dtype=torch.float32
        )

    def init_from_measurement(self, z0):
        z0 = torch.tensor(z0, dtype=torch.float32)
        self.x = torch.zeros(4, dtype=torch.float32)
        self.x[0:2] = z0
        self.x[2:4] = 0.0
        self.P = torch.eye(4) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + torch.eye(4) * self.q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + torch.eye(2) * self.r
        K = self.P @ self.H.T @ torch.inverse(S)
        self.x = self.x + K @ y
        self.P = (torch.eye(4) - K @ self.H) @ self.P

    def pos(self):
        return self.x[:2].clone()


def simulate_primitives_pattern_with_labels(pattern, seg_len=40, dt=0.1, v=2.6, omega=1.1, start_std=5.0, heading_init=0.0):
    p = torch.randn(2) * start_std
    heading = float(heading_init)
    pts = []
    labels = []

    for tok in pattern:
        if tok == "L":
            w = 0.0
        elif tok == "A+":
            w = abs(omega)
        elif tok == "A-":
            w = -abs(omega)
        else:
            raise ValueError(f"Unknown token: {tok}")

        for _ in range(seg_len):
            heading = heading + w * dt
            vx = v * math.cos(heading)
            vy = v * math.sin(heading)
            p = p + torch.tensor([vx, vy]) * dt
            pts.append(p.clone())
            labels.append(tok)

    return torch.stack(pts, dim=0), labels


def add_gaussian_noise(true_xy: np.ndarray, sigma: float) -> np.ndarray:
    return true_xy + np.random.randn(*true_xy.shape) * sigma


def make_sparse_mask(T: int, K: int, phase: int = 0) -> np.ndarray:
    m = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if ((t + phase) % K) == 0:
            m[t] = 1.0
    return m


def build_meas_in_hold_last(meas_full: np.ndarray, m: np.ndarray) -> np.ndarray:
    T = meas_full.shape[0]
    meas_in = np.zeros_like(meas_full)
    last = meas_full[0].copy()
    for t in range(T):
        if m[t] > 0.5:
            last = meas_full[t].copy()
        meas_in[t] = last
    return meas_in


def weighted_smoothl1(pred_pos, true_pos, weights, beta=0.5):
    diff = pred_pos - true_pos
    abs_diff = torch.abs(diff)
    beta_t = torch.tensor(beta, device=abs_diff.device, dtype=abs_diff.dtype)
    quad = torch.minimum(abs_diff, beta_t)
    lin = abs_diff - quad
    loss = 0.5 * quad * quad / beta + lin
    loss = loss.sum(dim=-1, keepdim=True)
    return (loss * weights).mean()


def loss_fn(pred_x, true_pos, mask, miss_w=4.5, vel_w=0.015):
    pred_pos = pred_x[:, :, 0:2]
    weights = 1.0 + (1.0 - mask) * (miss_w - 1.0)
    pos_loss = weighted_smoothl1(pred_pos, true_pos, weights, beta=0.5)

    pred_v = pred_x[:, 1:, 2:4]
    true_v = true_pos[:, 1:, :] - true_pos[:, :-1, :]
    vel_loss = ((pred_v - true_v) ** 2).mean()
    return pos_loss + vel_w * vel_loss


def rmse(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true) ** 2, axis=1))))


def max_err(pred, true):
    e = np.sqrt(np.sum((pred - true) ** 2, axis=1))
    return float(np.max(e))


def rmse_on_indices(pred, true, idx):
    if len(idx) == 0:
        return float("nan")
    d = pred[idx] - true[idx]
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def sample_train_pattern(n_segs=5):
    choices = ["L", "A+", "A-"]
    pat = [random.choice(choices) for _ in range(n_segs)]
    if not any(t in ["A+", "A-"] for t in pat):
        pat[random.randrange(len(pat))] = random.choice(["A+", "A-"])
    return pat


def apply_dropout_block_inside_arc(m, labels, block_len=24):
    T = len(labels)
    arc_idx = [i for i, tok in enumerate(labels) if tok in ["A+", "A-"]]
    if len(arc_idx) < block_len + 2:
        return m, None

    arc_set = set(arc_idx)
    candidates = []
    for s in arc_idx:
        ok = True
        for t in range(s, min(s + block_len, T)):
            if t not in arc_set:
                ok = False
                break
        if ok:
            candidates.append(s)

    if not candidates:
        return m, None

    s = random.choice(candidates)
    e = min(s + block_len, T)
    m2 = m.copy()
    m2[s:e] = 0.0
    return m2, (s, e)


def generate_train_batch(batch_size=32, K=5, n_segs=5, seg_len=35, dt=0.1, sigma=0.18,
                         p_block_dropout=0.6, block_len=26):
    true_batch, meas_in_batch, mask_batch = [], [], []

    for _ in range(batch_size):
        pat = sample_train_pattern(n_segs=n_segs)
        v = random.uniform(2.2, 3.2)
        omega = random.uniform(0.9, 1.5)

        true_t, labels = simulate_primitives_pattern_with_labels(
            pat, seg_len=seg_len, dt=dt, v=v, omega=omega, start_std=5.0, heading_init=0.0
        )
        true_np = true_t.numpy()
        meas_full = add_gaussian_noise(true_np, sigma=sigma)

        T = meas_full.shape[0]
        phase = random.randrange(K)
        m = make_sparse_mask(T, K=K, phase=phase)

        m[0] = 1.0

        if random.random() < p_block_dropout:
            m, _ = apply_dropout_block_inside_arc(m, labels, block_len=block_len)

        meas_in = build_meas_in_hold_last(meas_full, m)

        true_batch.append(torch.tensor(true_np, dtype=torch.float32))
        meas_in_batch.append(torch.tensor(meas_in, dtype=torch.float32))
        mask_batch.append(torch.tensor(m.reshape(T, 1), dtype=torch.float32))

    return (
        torch.stack(true_batch, dim=0),
        torch.stack(meas_in_batch, dim=0),
        torch.stack(mask_batch, dim=0),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    dt = 0.1
    K = 2
    sigma = 0.18
    train_epochs = 900
    block_len = 28

    # Loss weights
    miss_w = 4.5      # weight for frames with missing measurements (gap)
    vel_w  = 0.015    # velocity regularization weight
    omega_smooth_w = 0.2



    test_pattern = ["L", "A+", "L", "A-", "L"]
    test_seg_len = 36

    kalman_q = 1e-4

    print("Marketing demo (FIXED): sparse measurements + long dropout inside a turn.")
    print(f"K={K}, sigma={sigma}, dropout block_len={block_len}, Kalman q={kalman_q}")
    print(f"Test pattern: {test_pattern} (unseen combination)")

    model = LearnedSparseFilter(dt=dt, d_model=64, omega_cap=2.2)
    opt = torch.optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)

    model.train()
    for epoch in range(train_epochs):
        true_pos, meas_in, mask = generate_train_batch(
            batch_size=32, K=K, n_segs=5, seg_len=35, dt=dt, sigma=sigma,
            p_block_dropout=0.6, block_len=block_len
        )
        B, T, _ = meas_in.shape
        model.reset_stream(batch_size=B, dtype=torch.float32)

        # xs = []
        # for t in range(T):
        #     xs.append(model.step(meas_in[:, t, :], mask[:, t, :]))
        # pred_x = torch.stack(xs, dim=1)
        #
        # loss = loss_fn(pred_x, true_pos, mask, miss_w=3.5, vel_w=0.02)
        xs = []
        omegas = []

        for t in range(T):
            # --- Forward one step ---
            # NOTE: we need omega_t from the predictor to penalize jitter during gaps.
            # So we call pred_net directly with the same input used inside the model.
            z_t = meas_in[:, t, :]
            m_t = mask[:, t, :]

            # Run the model step (state update)
            x_t = model.step(z_t, m_t)
            xs.append(x_t)

            # Recompute omega with the same u_pred = [x, z, m] used for prediction
            # IMPORTANT: use the model's current internal state *before* step? We cannot.
            # Practical approximation: use x_t (after step). This still stabilizes omega.
            u_pred = torch.cat([x_t.detach(), z_t, m_t], dim=-1)  # [B,7]
            omega_t = model.pred_net.step(u_pred)  # [B,1]
            omegas.append(omega_t)

        pred_x = torch.stack(xs, dim=1)  # [B,T,4]
        omega_seq = torch.stack(omegas, dim=1)  # [B,T,1]

        # Base loss (position + optional velocity reg)
        loss_base = loss_fn(pred_x, true_pos, mask, miss_w=miss_w, vel_w=vel_w)

        # Smoothness loss on omega changes, ONLY when measurement is missing
        # Δomega_t = omega_t - omega_{t-1}, weight by (1 - m_t)
        domega = omega_seq[:, 1:, :] - omega_seq[:, :-1, :]  # [B,T-1,1]
        m_miss = 1.0 - mask[:, 1:, :]  # [B,T-1,1]
        loss_omega_smooth = (domega * domega * m_miss).mean()

        loss = loss_base + omega_smooth_w * loss_omega_smooth

        if torch.isnan(loss):
            print(f"Epoch {epoch}, Loss is NaN -> skipping update")
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Test trajectory
    model.eval()
    true_test_t, labels = simulate_primitives_pattern_with_labels(
        test_pattern, seg_len=test_seg_len, dt=dt, v=2.7, omega=1.25, start_std=5.0, heading_init=0.0
    )
    test_true = true_test_t.numpy()
    T = test_true.shape[0]

    meas_full = add_gaussian_noise(test_true, sigma=sigma)
    m = make_sparse_mask(T, K=K, phase=0)
    m[0] = 1.0

    m2, block = apply_dropout_block_inside_arc(m, labels, block_len=block_len)
    if block is None:
        s = T // 3
        e = min(s + block_len, T)
        m2 = m.copy()
        m2[s:e] = 0.0
        block = (s, e)

    m = m2
    gap_s, gap_e = block
    print(f"Test dropout block inside arc: frames [{gap_s}, {gap_e})")

    meas_in = build_meas_in_hold_last(meas_full, m)

    # Learned inference
    model.reset_stream(batch_size=1, dtype=torch.float32)
    learned_pred = []
    with torch.no_grad():
        for t in range(T):
            z_t = torch.tensor(meas_in[t:t+1], dtype=torch.float32)
            m_t = torch.tensor([[m[t]]], dtype=torch.float32)
            x_t = model.step(z_t, m_t)
            learned_pred.append(x_t[0, 0:2].numpy())
    learned_pred = np.array(learned_pred)

    # Kalman inference
    kf = KalmanCV(dt=dt, q=kalman_q, r=sigma**2)
    kf.init_from_measurement(meas_full[0])

    kf_pred = []
    for t in range(T):
        kf.predict()
        if m[t] > 0.5:
            kf.update(torch.tensor(meas_full[t], dtype=torch.float32))
        kf_pred.append(kf.pos().numpy())
    kf_pred = np.array(kf_pred)

    # Metrics
    err_l = np.sqrt(np.sum((learned_pred - test_true) ** 2, axis=1))
    err_k = np.sqrt(np.sum((kf_pred - test_true) ** 2, axis=1))
    idx_gap = list(range(gap_s, gap_e))

    print("\nTesting (sparse + long gap, streaming)...")
    print(f"RMSE total (Learned, Kalman): {rmse(learned_pred, test_true):.4f}, {rmse(kf_pred, test_true):.4f}")
    print(f"MaxErr total (Learned, Kalman): {max_err(learned_pred, test_true):.4f}, {max_err(kf_pred, test_true):.4f}")
    print(f"RMSE gap-window (Learned, Kalman): {rmse_on_indices(learned_pred, test_true, idx_gap):.4f}, {rmse_on_indices(kf_pred, test_true, idx_gap):.4f}")
    print(f"MaxErr gap-window (Learned, Kalman): {float(np.max(err_l[gap_s:gap_e])):.4f}, {float(np.max(err_k[gap_s:gap_e])):.4f}")
    print(f"Measurements available: {int(m.sum())} / {T} frames")

    # Plot XY
    plt.figure(figsize=(10, 6))
    plt.plot(test_true[:, 0], test_true[:, 1], "g-", linewidth=2, label="True Path")
    meas_idx = np.where(m > 0.5)[0]
    plt.scatter(meas_full[meas_idx, 0], meas_full[meas_idx, 1], s=28, alpha=0.85, label=f"Measurements (every {K} frames)")
    plt.plot(learned_pred[:, 0], learned_pred[:, 1], "b--", label="Learned (segments prior)")
    plt.plot(kf_pred[:, 0], kf_pred[:, 1], "k-.", linewidth=2, label="Kalman CV (over-confident)")

    start_xy = test_true[0]
    end_xy = test_true[-1]
    plt.scatter([start_xy[0]], [start_xy[1]], marker="*", s=200, label="True Start")
    plt.scatter([end_xy[0]], [end_xy[1]], marker="X", s=140, label="True End")
    plt.annotate("Start", (start_xy[0], start_xy[1]), textcoords="offset points", xytext=(8, 8))
    plt.annotate("End", (end_xy[0], end_xy[1]), textcoords="offset points", xytext=(8, 8))

    plt.title(f"Marketing demo (fixed): Sparse (K={K}) + long gap inside turn")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout()
    out_xy = "marketing_sparse_gap_inside_turn_xy_fixed.png"
    plt.savefig(out_xy)
    print(f"Saved: {out_xy}")

    # Plot error vs time
    t = np.arange(T)
    plt.figure(figsize=(10, 4))
    plt.plot(t, err_l, label="Learned error")
    plt.plot(t, err_k, label="Kalman error")
    plt.axvspan(gap_s, gap_e - 1, alpha=0.18, label="Forced measurement gap (inside turn)")
    plt.title("Position error vs time (gap highlighted)")
    plt.xlabel("Frame")
    plt.ylabel("Euclidean position error")
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    plt.tight_layout()
    out_err = "marketing_sparse_gap_inside_turn_error_fixed.png"
    plt.savefig(out_err)
    print(f"Saved: {out_err}")
