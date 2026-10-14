"""nanogpt: a character-level transformer written from scratch on numpy.

No framework, no autodiff, no dependency beyond numpy. Every forward and
backward pass is explicit so it can be read and checked against finite
differences with --gradcheck.
"""

import argparse
import json
import math
import sys
import time

import numpy as np

EPS = 1e-5
GC = 0.7978845608028654
GK = 0.044715


def vso_cfg():
    p = argparse.ArgumentParser(prog="nanogpt", description="char-level transformer from scratch")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--ctx", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--warmup", type=int, default=60)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--gen", type=int, default=400)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--topk", type=int, default=40)
    p.add_argument("--out", default="run.json")
    p.add_argument("--gradcheck", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p


def inf_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def inf_gelu(x):
    return 0.5 * x * (1.0 + np.tanh(GC * (x + GK * x ** 3)))


def inf_gelu_bwd(x, d):
    t = np.tanh(GC * (x + GK * x ** 3))
    du = GC * (1.0 + 3.0 * GK * x ** 2)
    return d * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * du)


def inf_ln_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    istd = 1.0 / np.sqrt(var + EPS)
    xh = xc * istd
    return xh * g + b, (xh, g, istd)


def inf_ln_bwd(dy, c):
    xh, g, istd = c
    D = xh.shape[-1]
    dg = (dy * xh).reshape(-1, D).sum(0)
    db = dy.reshape(-1, D).sum(0)
    dxh = dy * g
    dx = istd / D * (D * dxh - dxh.sum(-1, keepdims=True) - xh * (dxh * xh).sum(-1, keepdims=True))
    return dx, dg, db


def inf_outer(a, b):
    return np.tensordot(a, b, axes=([0, 1], [0, 1]))


def inf_outproj(a, d):
    return np.einsum("btv,btd->vd", d, a)


class vso_Tokenizer:
    def __init__(self, text):
        self.itos = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab = len(self.itos)

    def enc(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int32)

    def dec(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


class inf_Attention:
    def __init__(self, cfg, rng):
        D = cfg["dim"]
        self.hd = D // cfg["heads"]
        self.H = cfg["heads"]
        s = 1.0 / math.sqrt(self.hd)
        self.wq = rng.normal(0, s, (D, D))
        self.wk = rng.normal(0, s, (D, D))
        self.wv = rng.normal(0, s, (D, D))
        self.wo = rng.normal(0, s * 0.5, (D, D))
        self.kc = None
        self.vc = None
        self.use_cache = False
        self.c = None

    def fwd(self, x):
        B, T, D = x.shape
        H, hd = self.H, self.hd
        q = (x @ self.wq).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = (x @ self.wk).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        v = (x @ self.wv).reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        if self.use_cache:
            if self.kc is not None and self.kc.shape[2] > 0:
                k = np.concatenate([self.kc, k], axis=2)
                v = np.concatenate([self.vc, v], axis=2)
            self.kc, self.vc = k, v
        Tk = k.shape[2]
        p = inf_softmax(np.where(np.triu(np.ones((T, Tk), bool), k=Tk - T),
                                  -1e9, (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)), -1)
        y = (p @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        self.c = (x, q, k, v, p, T, Tk, y)
        return y @ self.wo, y

    def bwd(self, dout):
        x, q, k, v, p, T, Tk, y = self.c
        B, H, T, hd = q.shape
        D = H * hd
        dx = dout @ self.wo.T
        d_wo = inf_outer(y, dout)
        dy = dx.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        dp = dy @ v.transpose(0, 1, 3, 2)
        dv = p.transpose(0, 1, 3, 2) @ dy
        da = p * (dp - (dp * p).sum(-1, keepdims=True)) / math.sqrt(hd)
        da = np.where(np.triu(np.ones((T, Tk), bool), k=Tk - T), 0.0, da)
        dq = da @ k
        dk = da.transpose(0, 1, 3, 2) @ q
        u = lambda a: a.transpose(0, 2, 1, 3).reshape(B, T, D)
        dq, dk, dv = u(dq), u(dk), u(dv)
        dxs = dq @ self.wq.T + dk @ self.wk.T + dv @ self.wv.T
        return dxs, {"wq": inf_outer(x, dq), "wk": inf_outer(x, dk),
                     "wv": inf_outer(x, dv), "wo": d_wo}


class vso_Block:
    def __init__(self, cfg, rng):
        D = cfg["dim"]
        self.att = inf_Attention(cfg, rng)
        self.ln1g = np.ones(D)
        self.ln1b = np.zeros(D)
        self.ln2g = np.ones(D)
        self.ln2b = np.zeros(D)
        s = 1.0 / math.sqrt(D)
        self.fc1w = rng.normal(0, s, (D, 4 * D))
        self.fc1b = np.zeros(4 * D)
        self.fc2w = rng.normal(0, s * 0.5, (4 * D, D))
        self.fc2b = np.zeros(D)
        self.c = None

    def fwd(self, x):
        a, c1 = inf_ln_fwd(x, self.ln1g, self.ln1b)
        aout, y = self.att.fwd(a)
        x1 = x + aout
        b, c2 = inf_ln_fwd(x1, self.ln2g, self.ln2b)
        hp = b @ self.fc1w + self.fc1b
        h = inf_gelu(hp)
        x2 = x1 + h @ self.fc2w + self.fc2b
        self.c = (c1, c2, b, h, hp)
        return x2

    def bwd(self, dout):
        c1, c2, b, h, hp = self.c
        d_fc2w = inf_outer(h, dout)
        d_fc2b = dout.sum(axis=(0, 1))
        dh = inf_gelu_bwd(hp, dout @ self.fc2w.T)
        d_fc1w = inf_outer(b, dh)
        d_fc1b = dh.sum(axis=(0, 1))
        dx1, d_ln2g, d_ln2b = inf_ln_bwd(dh @ self.fc1w.T, c2)
        dx1 = dx1 + dout
        daout, ag = self.att.bwd(dx1)
        dxa, d_ln1g, d_ln1b = inf_ln_bwd(daout, c1)
        dx = dx1 + dxa
        return dx, {"ln1g": d_ln1g, "ln1b": d_ln1b, "ln2g": d_ln2g, "ln2b": d_ln2b,
                    "fc1w": d_fc1w, "fc1b": d_fc1b, "fc2w": d_fc2w, "fc2b": d_fc2b, **ag}


class inf_AdamW:
    def __init__(self, params, wd=0.01, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params
        self.wd = wd
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [np.zeros_like(x) for x in params]
        self.v = [np.zeros_like(x) for x in params]
        self.t = 0

    def step(self, grads, lr):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for i, (p, g) in enumerate(zip(self.p, grads)):
            if g is None:
                continue
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            u = (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)
            if p.ndim >= 2:
                u = u + self.wd * p
            p -= lr * u


class vso_Model:
    def __init__(self, vocab, cfg, rng):
        self.cfg = cfg
        self.vocab = vocab
        D = cfg["dim"]
        self.wte = rng.normal(0, 0.02, (vocab, D))
        self.wpe = rng.normal(0, 0.02, (cfg["ctx"], D))
        self.blocks = [vso_Block(cfg, rng) for _ in range(cfg["layers"])]
        self.lng = np.ones(D)
        self.lnb = np.zeros(D)
        self.c = None

    def named(self):
        n = [("wte", self.wte), ("wpe", self.wpe), ("lng", self.lng), ("lnb", self.lnb)]
        for i, b in enumerate(self.blocks):
            for k in ("ln1g", "ln1b", "ln2g", "ln2b", "fc1w", "fc1b", "fc2w", "fc2b"):
                n.append((f"b{i}.{k}", getattr(b, k)))
            for k in ("wq", "wk", "wv", "wo"):
                n.append((f"b{i}.{k}", getattr(b.att, k)))
        return n

    def params(self):
        return [v for _, v in self.named()]

    def fwd(self, idx, targets=None):
        B, T = idx.shape
        x = self.wte[idx] + self.wpe[:T]
        for b in self.blocks:
            x = b.fwd(x)
        xpre = x
        x, lc = inf_ln_fwd(x, self.lng, self.lnb)
        self.c = (idx, T, x, lc, xpre)
        logits = x @ self.wte.T
        if targets is None:
            return logits, None
        return logits, inf_ce(logits, targets)

    def bwd(self, dlogits):
        idx, T, x, lc, xpre = self.c
        d, d_lng, d_lnb = inf_ln_bwd(dlogits @ self.wte, lc)
        g = {"wte": inf_outproj(x, dlogits), "wpe": np.zeros_like(self.wpe), "lng": d_lng, "lnb": d_lnb}
        for i in range(len(self.blocks) - 1, -1, -1):
            d, bg = self.blocks[i].bwd(d)
            g.update({f"b{i}.{k}": v for k, v in bg.items()})
        for b_ in range(idx.shape[0]):
            np.add.at(g["wte"], idx[b_], d[b_])
        g["wpe"][:T] = d.sum(0)
        return g


def inf_ce(logits, targets):
    B, T, V = logits.shape
    p = inf_softmax(logits, -1).reshape(-1, V)
    y = targets.reshape(-1)
    n = max(y.shape[0], 1)
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
    d = p.copy()
    d[np.arange(n), y] -= 1.0
    return float(loss), (d / n).reshape(B, T, V)


def inf_lr(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    r = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["minlr"] + 0.5 * (cfg["lr"] - cfg["minlr"]) * (1 + math.cos(math.pi * min(r, 1.0)))


def inf_gradcheck(cfg, seed=0):
    rng = np.random.default_rng(seed)
    vocab, B, T = 11, 3, 5
    m = vso_Model(vocab, cfg, rng)
    idx = rng.integers(0, vocab, (B, T))
    y = rng.integers(0, vocab, (B, T))
    tgt = y[:, 1:]
    inp = idx[:, :-1]

    def loss_of():
        _, (l, _) = m.fwd(inp, tgt)
        return l

    _, (_, dlogits) = m.fwd(inp, tgt)
    g = m.bwd(dlogits)
    worst = 0.0
    for name, p in m.named():
        flat = p.reshape(-1)
        idxs = rng.choice(flat.shape[0], size=min(6, flat.shape[0]), replace=False)
        for i in idxs:
            o = flat[i]
            h = 1e-4
            flat[i] = o + h
            lp = loss_of()
            flat[i] = o - h
            lm = loss_of()
            flat[i] = o
            num = (lp - lm) / (2 * h)
            ana = g[name].reshape(-1)[i]
            den = max(1e-8, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / den)
    return worst


class vso_Gen:
    def __init__(self, model, tok):
        self.m = model
        self.tok = tok

    def reset(self):
        for b in self.m.blocks:
            b.att.use_cache = True
            b.att.kc = np.zeros((1, b.att.H, 0, b.att.hd))
            b.att.vc = np.zeros((1, b.att.H, 0, b.att.hd))

    def off(self):
        for b in self.m.blocks:
            b.att.use_cache = False
            b.att.kc = None
            b.att.vc = None

    def sample(self, idx, n, temp, topk, rng):
        self.reset()
        ctx = self.m.cfg["ctx"]
        for _ in range(n):
            inp = idx[:, -ctx:]
            logits, _ = self.m.fwd(inp)
            l = logits[:, -1, :] / max(temp, 1e-3)
            if topk > 0:
                k = min(topk, l.shape[-1])
                cut = np.argpartition(l, -k, axis=-1)[:, :k]
                l = np.take_along_axis(l, cut, -1)
                p = inf_softmax(l, -1)
                s = np.zeros_like(l)
                np.put_along_axis(s, cut, p, -1)
            else:
                s = inf_softmax(l, -1)
            nxt = np.array([rng.choice(len(r_), p=r_) for r_ in s])[:, None]
            idx = np.concatenate([idx, nxt.astype(idx.dtype)], axis=1)
        return idx

CORPUS = """
the machine learns the shape of the data before it learns the name of the data
a model is a function fitted to data and judged by the data it never saw
gradient descent is a walk down a hill you can only feel with your feet
attention lets a sequence look at its own past and decide what matters now
the loss goes down slowly and then suddenly and that is the whole of training
weight decay keeps the model humble so it does not memorize what it was given
a tokenizer turns letters into integers because numbers are what tensors eat
overfitting is the model learning the noise and calling it a pattern
the learning rate decides whether you arrive or you overshoot into the void
cross entropy measures surprise and surprise is what we call learning here
softmax turns scores into probabilities and probabilities sum to one
the residual connection lets the gradient find a path home through the deep
a transformer without positional encoding cannot tell first from second
temperature scales the logits and low temperature makes the model careful
knowledge distillation lets a small student learn from a large teacher
normalization keeps the activations in a range the optimizer can handle
"""


def vso_batches(data, B, T, rng, n):
    hi = len(data) - T - 1
    for _ in range(n):
        ix = rng.integers(0, hi, B)
        x = np.stack([data[i:i + T] for i in ix])
        y = np.stack([data[i + 1:i + 1 + T] for i in ix])
        yield x, y


def vso_train(cfg, text, seed=1337, quiet=False):
    rng = np.random.default_rng(seed)
    tok = vso_Tokenizer(text)
    data = tok.enc(text)
    m = vso_Model(tok.vocab, cfg, rng)
    names = [n for n, _ in m.named()]
    ps = m.params()
    opt = inf_AdamW(ps, wd=0.01)
    gen = vso_Gen(m, tok)
    T = cfg["ctx"]
    hist = []
    t0 = time.time()
    for step, (x, y) in enumerate(vso_batches(data, cfg["batch"], T, rng, cfg["steps"])):
        _, (loss, dl) = m.fwd(x, y)
        g = m.bwd(dl)
        gl = [g[n] for n in names]
        tot = np.sqrt(sum((v ** 2).sum() for v in gl))
        if cfg["clip"] > 0 and tot > cfg["clip"]:
            sc = cfg["clip"] / (tot + 1e-6)
            gl = [v * sc for v in gl]
        lr = inf_lr(step, cfg)
        opt.step(gl, lr)
        hist.append(loss)
        if not quiet and (step % max(1, cfg["steps"] // 10) == 0 or step == cfg["steps"] - 1):
            print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | gnorm {tot:.3f}")
    gen.off()
    return m, tok, gen, hist, time.time() - t0


def main(argv=None):
    a = vso_cfg().parse_args(argv)
    if a.dim % a.heads:
        sys.stderr.write("dim must be divisible by heads\n")
        return 2
    cfg = {"dim": a.dim, "heads": a.heads, "layers": a.layers, "ctx": a.ctx,
           "lr": a.lr, "steps": a.steps, "batch": a.batch, "warmup": a.warmup,
           "clip": a.clip, "minlr": a.lr * 0.1, "dropout": a.dropout, "seed": a.seed}
    if a.gradcheck:
        w = inf_gradcheck(cfg)
        print(f"gradcheck worst relative error: {w:.3e}")
        print("PASS" if w < 2e-3 else "FAIL")
        return 0 if w < 2e-3 else 1
    m, tok, gen, hist, el = vso_train(cfg, CORPUS, a.seed, a.quiet)
    npar = sum(p.size for p in m.params())
    first, last = hist[0], hist[-1]
    res = {"params": npar, "vocab": tok.vocab, "layers": a.layers, "dim": a.dim,
           "heads": a.heads, "ctx": a.ctx, "steps": a.steps,
           "loss_first": round(first, 4), "loss_last": round(last, 4),
           "loss_curve": [round(x, 4) for x in hist],
           "seconds": round(el, 2), "params_per_sec": int(a.steps / el) if el else 0}
    if a.gen > 0:
        rng = np.random.default_rng(a.seed)
        prompt = np.array([[tok.stoi[c] for c in "the machine"]], dtype=np.int32)
        out = gen.sample(prompt, a.gen, a.temp, a.topk, rng)
        text = tok.dec(out[0][-a.gen:])
        res["sample"] = text
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    if not a.quiet:
        print(f"\nparams {npar} | vocab {tok.vocab} | {el:.1f}s")
        if a.gen > 0:
            print("\nsample:\n" + res["sample"])
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
