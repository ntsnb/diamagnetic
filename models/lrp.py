from __future__ import annotations

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn

from models.attention_blocks import SimpleMultiheadAttention

class LRPEngine:
    """LRP helper.

    Exposes two interfaces:
      - compute_relevance_map: offline discrepancy regularizer
      - compute_drift: Stage II online drift
    """

    def __init__(self) -> None:
        super().__init__()

    def compute_relevance_map(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int] = None,
        method: str = "rrp",
        eps: float = 1e-6,
        gradient_modulation: bool = True,
        return_all_layers: bool = False,
        forward_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute relevance aligned with input.

        method:
          - "rrp": regression relevance propagation (CausalFormer Eq. 17/19)
          - "epsilon": LRP-ε for Linear layers; fallback to grad*input otherwise.
          - "grad": grad*input
        """

        if method == "rrp":
            return self._rrp(
                model,
                x,
                target_idx,
                eps,
                gradient_modulation,
                return_all_layers=return_all_layers,
                forward_kwargs=forward_kwargs,
            )
        if method == "grad":
            return self._grad_input(model, x, target_idx, forward_kwargs=forward_kwargs)
        return self._lrp_epsilon(model, x, target_idx, eps, forward_kwargs=forward_kwargs)

    # --------- Convenience APIs ---------
    def get_input_relevance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int] = None,
    ) -> torch.Tensor:
        out = self.compute_relevance_map(
            model, x, target_idx, method="rrp", return_all_layers=False
        )
        return out

    def get_layer_relevance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.compute_relevance_map(
            model, x, target_idx, method="rrp", return_all_layers=True
        )
        return out["layers"]

    def get_qkv_relevance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        return self.compute_attention_logits_relevance(model, x, target_idx)

    def compute_drift(
        self,
        x: torch.Tensor,
        model_cur: nn.Module,
        model_ref: nn.Module,
        target_idx: Optional[int] = None,
        method: str = "rrp",
        forward_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        r_cur = self.compute_relevance_map(
            model_cur, x, target_idx, method=method, forward_kwargs=forward_kwargs
        )
        r_ref = self.compute_relevance_map(
            model_ref, x, target_idx, method=method, forward_kwargs=forward_kwargs
        )
        return r_cur - r_ref

    def _select_target(self, y: torch.Tensor, target_idx: Optional[int]) -> torch.Tensor:
        # y: [B, ...]
        if hasattr(y, "y_hat"):
            y = y.y_hat
        if target_idx is None:
            return y.sum()
        # assume last dim indexes forecast horizon or output dim
        return y[..., target_idx].sum()

    def _grad_input(
        self, model: nn.Module, x: torch.Tensor, target_idx: Optional[int], forward_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        x = x.clone().detach().requires_grad_(True)
        y = model(x, **(forward_kwargs or {}))
        target = self._select_target(y, target_idx)
        target.backward()
        return x * x.grad

    def _lrp_epsilon(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int],
        eps: float,
        forward_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """LRP-ε for Linear layers; fallback to grad*input if unsupported."""

        # If model contains unsupported modules, fallback to grad*input.
        supported = all(
            isinstance(m, (nn.Linear, nn.ReLU, nn.LayerNorm, nn.Sequential))
            or m is model
            for m in model.modules()
        )
        if not supported:
            return self._grad_input(model, x, target_idx)

        # Capture activations for Linear layers.
        activations = {}
        outputs = {}

        def hook_fn(name):
            def _hook(m, inp, out):
                activations[name] = inp[0].detach()
                outputs[name] = out.detach()
            return _hook

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(hook_fn(name)))

        x = x.clone().detach().requires_grad_(True)
        y = model(x, **(forward_kwargs or {}))
        for h in hooks:
            h.remove()

        # Initialize relevance at output
        if y.dim() == 1:
            r = y.clone()
        else:
            r = torch.zeros_like(y)
            if target_idx is None:
                r = y
            else:
                r[..., target_idx] = y[..., target_idx]

        # Propagate relevance backward through Linear layers
        for name, m in reversed(list(model.named_modules())):
            if not isinstance(m, nn.Linear):
                continue
            a = activations[name]
            z = outputs[name]
            w = m.weight  # [out, in]
            # z: [B, out], a: [B, in]
            z_stable = z + eps * torch.sign(z)
            # compute contribution
            # [B, in, out] = [B, in, 1] * [1, in, out]
            contrib = a.unsqueeze(2) * w.t().unsqueeze(0)
            # normalize over inputs
            norm = z_stable.unsqueeze(1)
            message = contrib / norm
            r = (message * r.unsqueeze(1)).sum(dim=2)

        # r now aligned to input of first linear; if model starts with linear, map to x
        if r.shape != x.shape:
            # fallback to grad*input if shape mismatch
            return self._grad_input(model, x, target_idx)
        return r

    def _rrp(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int],
        eps: float,
        gradient_modulation: bool,
        return_all_layers: bool = False,
        forward_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Regression Relevance Propagation (CausalFormer Eq. 17/19).

        R_i = sum_j x_i * (∂f_j/∂x_i) * (R_j / f_j(x))
        If target_idx is provided, set R_j = 1 for that target; else R_j = f_j(x).
        """

        # enable per-layer capture if requested
        if return_all_layers:
            for m in model.modules():
                if hasattr(m, "capture_grads"):
                    m.capture_grads = True

        x = x.clone().detach().requires_grad_(True)
        y = model(x, **(forward_kwargs or {}))
        # FactorMechanism returns FactorOutputs; extract y_hat if needed
        if hasattr(y, "y_hat"):
            y = y.y_hat

        if target_idx is None:
            # R_j = f_j(x) => R_i = x_i * ∂(sum_j f_j)/∂x_i
            s = y.sum()
            s.backward()
            g = x.grad
            r = x * g
        else:
            # One-hot target relevance
            y_t = y[..., target_idx]
            s = y_t.sum()
            s.backward()
            g = x.grad
            denom = y_t.detach().unsqueeze(-1).unsqueeze(-1)
            while denom.dim() < x.dim():
                denom = denom.unsqueeze(-1)
            r = x * g / (denom + eps)

        if gradient_modulation:
            r = torch.relu(x.grad) * r

        if not return_all_layers:
            # reset flags
            for m in model.modules():
                if hasattr(m, "capture_grads"):
                    m.capture_grads = False
            return r

        # collect per-layer relevance (grad*activation)
        layer_scores: Dict[str, torch.Tensor] = {}
        for name, m in model.named_modules():
            if hasattr(m, "last_out") and m.last_out is not None and m.last_out.grad is not None:
                layer_scores[name] = m.last_out * m.last_out.grad

        for m in model.modules():
            if hasattr(m, "capture_grads"):
                m.capture_grads = False

        return {"input": r, "layers": layer_scores}

    def attention_relevance(
        self, attn: torch.Tensor, v: torch.Tensor, r_out: torch.Tensor, eps: float = 1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose relevance for matrix product (Eq. 18).

        attn: [B, H, T, T], v: [B, H, T, D], r_out: [B, H, T, D]
        Returns relevance for attn and v.
        """

        # R_out for A*V, distribute to A and V
        # R_A (B,H,T,T) and R_V (B,H,T,D)
        b, h, t, d = v.shape
        r_a = torch.zeros_like(attn)
        r_v = torch.zeros_like(v)

        # For each output position, relevance share is proportional to input contribution
        # A*V: out[t] = sum_k A[t,k] * V[k]
        # We use simple proportional rule
        v_sum = v.abs().sum(dim=-1, keepdim=True) + eps  # [B,H,T,1]
        r_v = r_out * (v.abs() / v_sum)

        # For attention, aggregate relevance from output and V
        v_mag = v.abs().sum(dim=-1) + eps  # [B,H,T]
        # distribute per key position
        r_a = (r_out.abs().sum(dim=-1).unsqueeze(-1) * (v_mag.unsqueeze(-2) / (v_mag.sum(dim=-1, keepdim=True) + eps)))
        return r_a, r_v

    def compute_attention_logits_relevance(
        self,
        model: nn.Module,
        x: torch.Tensor,
        target_idx: Optional[int] = None,
        forward_kwargs: Optional[dict] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute relevance on attention logits (QK^T) and Q/K/V.

        Returns a list per MHA module with keys:
          - "attn_scores", "q", "k", "v"
        """

        # enable capture
        mhas: List[SimpleMultiheadAttention] = []
        for m in model.modules():
            if isinstance(m, SimpleMultiheadAttention):
                m.capture_grads = True
                mhas.append(m)

        x = x.clone().detach().requires_grad_(True)
        y = model(x, **(forward_kwargs or {}))
        target = self._select_target(y, target_idx)
        target.backward()

        results: List[Dict[str, torch.Tensor]] = []
        for m in mhas:
            if m.last_attn_scores is None:
                continue
            # Gradient modulation per CausalFormer Eq.19
            attn_scores = m.last_attn_scores
            q = m.last_q
            k = m.last_k
            v = m.last_v
            rel_attn = attn_scores * attn_scores.grad if attn_scores.grad is not None else attn_scores
            rel_q = q * q.grad if q is not None and q.grad is not None else q
            rel_k = k * k.grad if k is not None and k.grad is not None else k
            rel_v = v * v.grad if v is not None and v.grad is not None else v
            results.append(
                {
                    "attn_scores": rel_attn,
                    "q": rel_q,
                    "k": rel_k,
                    "v": rel_v,
                }
            )
            m.capture_grads = False
        return results

    def compute_attention_head_scores(
        self, model: nn.Module, x: torch.Tensor, target_idx: Optional[int] = None, forward_kwargs: Optional[dict] = None
    ) -> Dict[int, torch.Tensor]:
        """Return per-head relevance scores for each MHA module.

        Returns dict: module_id -> scores [H]
        """

        rel = self.compute_attention_logits_relevance(model, x, target_idx, forward_kwargs=forward_kwargs)
        scores: Dict[int, torch.Tensor] = {}
        for i, item in enumerate(rel):
            attn_scores = item.get("attn_scores")
            if attn_scores is None:
                continue
            # attn_scores: [B, H, T, T]
            s = attn_scores.abs().mean(dim=(0, 2, 3))  # [H]
            scores[i] = s
        return scores

    def compute_attention_causal_scores(
        self, model: nn.Module, x: torch.Tensor, target_idx: Optional[int] = None, forward_kwargs: Optional[dict] = None
    ) -> Dict[int, torch.Tensor]:
        """Compute S(A) style causal scores from attention logits relevance.

        Returns dict: module_id -> S(A) [B, H, T, T]
        """

        rel = self.compute_attention_logits_relevance(model, x, target_idx, forward_kwargs=forward_kwargs)
        scores: Dict[int, torch.Tensor] = {}
        for i, item in enumerate(rel):
            attn_scores = item.get("attn_scores")
            if attn_scores is None:
                continue
            # Positive relevance only (Eq.19)
            scores[i] = torch.relu(attn_scores)
        return scores
