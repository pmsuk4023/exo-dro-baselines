"""KL-DRO inner optimization for a uniform empirical distribution."""

import math

import torch


def _kl_to_uniform(weights: torch.Tensor) -> torch.Tensor:
    """Return D_KL(weights || Uniform(n))."""
    n_samples = weights.numel()
    positive_weights = weights[weights > 0]
    return torch.sum(
        positive_weights
        * (torch.log(positive_weights) + math.log(n_samples))
    )


def _tilted_weights(losses: torch.Tensor, eta: float) -> torch.Tensor:
    """Compute exponential-tilted weights without direct exp overflow."""
    logits = losses / eta
    return torch.softmax(logits, dim=0)


def worst_case_weights(losses: torch.Tensor, eps: float) -> torch.Tensor:
    """Find worst-case weights under KL(q || Uniform(n)) <= eps.

    The dual solution has q_i proportional to exp(loss_i / eta).  Since the
    KL divergence of this distribution decreases with eta, eta is found by
    bracketing and bisection.
    """
    if not isinstance(losses, torch.Tensor):
        raise TypeError("losses must be a torch.Tensor")
    if losses.ndim != 1 or losses.numel() == 0:
        raise ValueError("losses must be a non-empty 1-D tensor")
    if not torch.is_floating_point(losses):
        raise TypeError("losses must have a floating-point dtype")
    if not math.isfinite(float(eps)) or eps < 0:
        raise ValueError("eps must be a finite non-negative number")
    if not torch.isfinite(losses).all():
        raise ValueError("losses must contain only finite values")

    # Double precision gives the KL feasibility check useful numerical margin.
    work_losses = losses.detach().to(dtype=torch.float64)
    n_samples = work_losses.numel()
    uniform = torch.full_like(work_losses, 1.0 / n_samples)

    if eps == 0:
        return uniform

    max_kl = math.log(n_samples)
    if eps >= max_kl:
        weights = torch.zeros_like(work_losses)
        weights[torch.argmax(work_losses)] = 1.0
        return weights

    def kl_at(eta: float) -> float:
        return float(_kl_to_uniform(_tilted_weights(work_losses, eta)))

    # Find an upper eta whose KL divergence is below the requested radius.
    lower_eta = 0.0
    upper_eta = max(1.0, float(torch.max(torch.abs(work_losses))))
    while kl_at(upper_eta) > eps:
        upper_eta *= 2.0

    # KL(q_eta || p) is monotone decreasing in eta for eta > 0.
    for _ in range(100):
        middle_eta = (lower_eta + upper_eta) / 2.0
        if middle_eta == 0.0:
            middle_eta = torch.finfo(work_losses.dtype).tiny
        if kl_at(middle_eta) > eps:
            lower_eta = middle_eta
        else:
            upper_eta = middle_eta

    weights = _tilted_weights(work_losses, upper_eta)
    kl_value = _kl_to_uniform(weights)
    tolerance = 1e-10 * max(1.0, eps)
    if float(kl_value) > eps + tolerance:
        raise RuntimeError(
            f"KL constraint violated: {float(kl_value):.12g} > {eps:.12g}"
        )
    return weights


def _run_tests() -> None:
    losses = torch.tensor([0.1, 0.2, 1.5, 0.1])
    eps_values = [0.0, 0.01, 0.1, 0.5, 2.0]
    previous_high_loss_weight = -1.0

    for eps in eps_values:
        weights = worst_case_weights(losses, eps)
        kl_value = float(_kl_to_uniform(weights))
        high_loss_weight = float(weights[2])

        assert torch.all(weights >= 0)
        assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype))
        assert kl_value <= eps + 1e-10 * max(1.0, eps)
        assert high_loss_weight >= previous_high_loss_weight
        previous_high_loss_weight = high_loss_weight

        print(
            f"eps={eps:>4.2f} | weights={weights.tolist()} "
            f"| KL={kl_value:.8f}"
        )


if __name__ == "__main__":
    _run_tests()
