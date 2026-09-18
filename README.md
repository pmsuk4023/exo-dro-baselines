# W-DRO & KL-DRO Baselines

baseline 비교 실험을 위한
Wasserstein DRO (W-DRO)와 KL-DRO 구현입니다.

공통 prediction model은 다음 MLP 구조를 사용합니다.

- Input: 4
- Hidden: 64 → 64
- Activation: ReLU
- Output: 1
- Loss: MSE

## W-DRO

Reference:
https://github.com/namkoong-lab/dro

`namkoong-lab/dro`의 neural W-DRO 구현을 참고했습니다.

- Robustness parameter: epsilon
- L2 adversarial perturbation
- Inner maximization을 통해 adversarial input 생성
- 생성된 adversarial input으로 공통 MLP 학습

Implementation:
`baselines/wdro.py`

## KL-DRO

공개 KL-DRO 구현은 linear model + CVXPY 기반이므로,
공통 MLP regression에 적용하기 위해 neural regression 형태로 구현했습니다.

- Robustness parameter: epsilon
- Mini-batch sample loss 계산
- KL divergence constraint 내 worst-case sample weights 계산
- Worst-case weight를 적용한 MSE로 MLP 학습

Implementation:
`baselines/kldro.py`

KL inner optimization test:
`experiments/test_kldro_inner.py`

## Hyperparameter Sweep

- W-DRO: `experiments/sweep_wdro.py`
- KL-DRO: `experiments/sweep_kldro.py`

Validation/OOD protocol 및 최종 hyperparameter 설정은
전체 실험 설정 확정 후 적용할 예정입니다.
