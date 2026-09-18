from baselines.kldro import KLDRO


def sweep_kldro(
    train_loader,
    val_loaders,
    eps_candidates,
    epochs=100,
    lr=1e-3
):
    """Train and compare KL-DRO models across validation environments.

    Parameters
    ----------
    train_loader:
        Common training DataLoader.

    val_loaders:
        Validation OOD DataLoader objects, one for each environment.

    eps_candidates:
        KL ambiguity-set radii to evaluate.

    epochs:
        Number of training epochs for each candidate.

    lr:
        Adam learning rate for each candidate model.

    Returns
    -------
    best_eps:
        Candidate with the lowest average validation MSE.

    results:
        Per-candidate validation results containing ``eps``, ``val_mses``,
        and ``average_val_mse``.
    """
    val_loaders = list(val_loaders)
    eps_candidates = list(eps_candidates)

    if not val_loaders:
        raise ValueError("val_loaders must contain at least one DataLoader")
    if not eps_candidates:
        raise ValueError("eps_candidates must contain at least one value")

    results = []

    for eps in eps_candidates:
        print("=" * 60)
        print(f"Testing eps = {eps}")
        print("=" * 60)

        model = KLDRO(
            eps=eps,
            lr=lr
        )
        model.train(
            train_loader,
            epochs=epochs
        )

        val_mses = []
        for val_loader in val_loaders:
            result = model.evaluate(val_loader)
            val_mses.append(result["mse"])

        average_val_mse = sum(val_mses) / len(val_mses)

        results.append(
            {
                "eps": eps,
                "val_mses": val_mses,
                "average_val_mse": average_val_mse,
            }
        )

        print(
            f"eps={eps}, "
            f"Validation OOD MSE={average_val_mse:.6f}"
        )

    # 최종 validation protocol이 정해질 때까지 평균 validation MSE가
    # 가장 작은 eps를 선택하는 임시 기준을 사용한다.
    best_result = min(
        results,
        key=lambda result: result["average_val_mse"]
    )
    best_eps = best_result["eps"]

    print()
    print("=" * 60)
    print("KL-DRO Validation Sweep Finished")
    print("=" * 60)

    for result in results:
        print(
            f"eps={result['eps']} "
            f"| val_mses={result['val_mses']} "
            f"| average_val_mse={result['average_val_mse']:.6f}"
        )

    print()
    print(f"Best eps: {best_eps}")

    return best_eps, results
