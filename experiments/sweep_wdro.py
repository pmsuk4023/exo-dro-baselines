from baselines.wdro import WassersteinDRO


def sweep_wdro(
    train_loader,
    val_loaders,
    epsilon_candidates,
    epochs=100,
    lr=1e-3,
    inner_steps=10,
    inner_lr=0.02
):
    """
    여러 epsilon으로 W-DRO를 학습한 뒤,
    Validation OOD MSE가 가장 작은 epsilon을 선택한다.

    Parameters
    ----------
    train_loader:
        공통 Training DataLoader

    val_loaders:
        Validation OOD DataLoader들의 리스트

    epsilon_candidates:
        시험할 epsilon 후보들

    Returns
    -------
    best_epsilon:
        Validation OOD에서 가장 좋은 epsilon

    results:
        epsilon별 Validation OOD MSE 결과
    """

    results = []

    # ---------------------------------------------------------
    # epsilon 후보를 하나씩 시험
    # ---------------------------------------------------------
    for epsilon in epsilon_candidates:

        print("=" * 60)
        print(f"Testing epsilon = {epsilon}")
        print("=" * 60)

        # -----------------------------------------------------
        # 1. 해당 epsilon으로 새로운 W-DRO 생성
        # -----------------------------------------------------
        model = WassersteinDRO(
            epsilon=epsilon,
            lr=lr,
            inner_steps=inner_steps,
            inner_lr=inner_lr
        )

        # -----------------------------------------------------
        # 2. 동일한 Training Data로 학습
        # -----------------------------------------------------
        model.train(
            train_loader,
            epochs=epochs
        )

        # -----------------------------------------------------
        # 3. Validation OOD 환경들에서 평가
        # -----------------------------------------------------
        val_mses = []

        for val_loader in val_loaders:

            result = model.evaluate(
                val_loader
            )

            val_mses.append(
                result["mse"]
            )

        # -----------------------------------------------------
        # 4. Validation OOD 평균 MSE
        # -----------------------------------------------------
        average_val_mse = (
            sum(val_mses) / len(val_mses)
        )

        print(
            f"epsilon={epsilon}, "
            f"Validation OOD MSE={average_val_mse:.6f}"
        )

        # -----------------------------------------------------
        # 5. 결과 저장
        # -----------------------------------------------------
        results.append(
            {
                "epsilon": epsilon,
                "val_mse": average_val_mse
            }
        )

    # ---------------------------------------------------------
    # 6. Validation MSE가 가장 작은 epsilon 선택
    # ---------------------------------------------------------
    best_result = min(
        results,
        key=lambda x: x["val_mse"]
    )

    best_epsilon = best_result["epsilon"]

    print()
    print("=" * 60)
    print("W-DRO Validation Sweep Finished")
    print("=" * 60)

    for result in results:
        print(
            f"epsilon={result['epsilon']} "
            f"| val_mse={result['val_mse']:.6f}"
        )

    print()
    print(
        f"Best epsilon: {best_epsilon}"
    )

    return best_epsilon, results