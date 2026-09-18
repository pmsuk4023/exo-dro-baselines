import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from common.mlp import build_mlp
from common.scm import forward_scm


class WassersteinDRO:
    def __init__(
        self,
        epsilon=0.1,
        lr=1e-3,
        inner_steps=10,
        inner_lr=0.02
    ):


        if epsilon < 0:
            raise ValueError(
                f"epsilon must be >= 0, got {epsilon}"
        )

        if inner_steps < 1:
            raise ValueError(
                f"inner_steps must be >= 1, got {inner_steps}"
        )

        if inner_lr <= 0:
            raise ValueError(
                f"inner_lr must be > 0, got {inner_lr}"
        )

        
        # 모든 baseline에서 동일하게 사용하는 공통 MLP
        self.model = build_mlp()

        # W-DRO robustness hyperparameter
        self.epsilon = epsilon

        # Inner maximization 설정
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr

        # GPU 사용 가능하면 GPU, 아니면 CPU 사용
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Regression이므로 MSE 사용
        # sample별 loss를 유지하기 위해 reduction="none"
        self.criterion = nn.MSELoss(reduction="none")

        # MLP parameter 학습용 optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

    # =========================================================
    # W-DRO Inner Maximization
    # =========================================================
    def inner_maximization(self, x, y):
        # 원본 x와 별도로 worst-case 입력 생성
        x_adv = x.detach().clone()
        x_adv.requires_grad_(True)

        for _ in range(self.inner_steps):
            # 현재 adversarial input에서 예측
            predictions = self.model(x_adv)

            # MSE loss
            loss = self.criterion(predictions, y).mean()

            # loss를 증가시키는 x 방향의 gradient 계산
            grad = torch.autograd.grad(
                loss,
                x_adv
            )[0]

            # 각 sample의 gradient를 L2 normalization
            grad_norm = torch.norm(
                grad,
                p=2,
                dim=1,
                keepdim=True
            )

            normalized_grad = grad / (grad_norm + 1e-12)

            with torch.no_grad():
                # loss가 증가하는 방향으로 이동
                x_adv = (
                    x_adv
                    + self.inner_lr * normalized_grad
                )

                # 원본 x로부터의 perturbation
                delta = x_adv - x

                # perturbation의 L2 거리
                delta_norm = torch.norm(
                    delta,
                    p=2,
                    dim=1,
                    keepdim=True
                )

                # epsilon L2-ball 안으로 projection
                factor = torch.clamp(
                    self.epsilon / (delta_norm + 1e-12),
                    max=1.0
                )

                x_adv = x + delta * factor

            # 다음 inner step에서 다시 gradient 계산
            x_adv.requires_grad_(True)

        return x_adv.detach()

    # =========================================================
    # Common Interface: train
    # =========================================================
    def train(self, train_loader, epochs=100):
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for x, y in train_loader:
                # 데이터를 GPU 또는 CPU로 이동
                x = x.to(self.device)
                y = y.to(self.device)

                # 1. 현재 모델에서 worst-case input 생성
                x_adv = self.inner_maximization(x, y)

                # 2. 이전 gradient 초기화
                self.optimizer.zero_grad()

                # 3. worst-case input으로 예측
                predictions = self.model(x_adv)

                # 4. robust training loss 계산
                loss = self.criterion(
                    predictions,
                    y
                ).mean()

                # 5. MLP parameter에 대한 gradient 계산
                loss.backward()

                # 6. MLP parameter 업데이트
                self.optimizer.step()

                epoch_loss += (
                    loss.item() * x.size(0)
                )

            epoch_loss /= len(train_loader.dataset)

            # 10 epoch마다 loss 출력
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Loss: {epoch_loss:.6f}"
                )

    # =========================================================
    # Common Interface: predict
    # =========================================================
    def predict(self, data_loader):
        self.model.eval()

        all_predictions = []

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)

                predictions = self.model(x)

                # 결과는 CPU로 이동해서 저장
                all_predictions.append(
                    predictions.cpu()
                )

        return torch.cat(
            all_predictions,
            dim=0
        )

    # =========================================================
    # Common Interface: evaluate
    # =========================================================
    def evaluate(self, data_loader):
        self.model.eval()

        total_squared_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)

                squared_error = (
                    predictions - y
                ) ** 2

                total_squared_error += (
                    squared_error.sum().item()
                )

                total_samples += y.numel()

        mse = (
            total_squared_error
            / total_samples
        )

        return {
            "mse": mse
        }

    # =========================================================
    # Common Interface: save
    # =========================================================
    def save(self, path):
        torch.save(
            {
                "model_state_dict":
                    self.model.state_dict(),

                "epsilon":
                    self.epsilon,

                "inner_steps":
                    self.inner_steps,

                "inner_lr":
                    self.inner_lr,
            },
            path
        )

    # =========================================================
    # Common Interface: load
    # =========================================================
    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.epsilon = checkpoint["epsilon"]
        self.inner_steps = checkpoint["inner_steps"]
        self.inner_lr = checkpoint["inner_lr"]

        self.model.to(self.device)


# =============================================================
# Smoke Test
# =============================================================
if __name__ == "__main__":

    # 재현 가능한 테스트를 위한 seed
    torch.manual_seed(0)

    # ---------------------------------------------------------
    # 1. Nominal exogenous variables U 생성
    # ---------------------------------------------------------
    n_samples = 1000

    U1 = torch.randn(n_samples)
    U2 = torch.randn(n_samples)
    U3 = torch.randn(n_samples)
    U4 = torch.randn(n_samples)
    Uy = torch.randn(n_samples)

    # ---------------------------------------------------------
    # 2. 교수님 SCM을 통해 X, y 생성
    # ---------------------------------------------------------
    X, y = forward_scm(
        U1,
        U2,
        U3,
        U4,
        Uy
    )

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # ---------------------------------------------------------
    # 3. Dataset + DataLoader 생성
    # ---------------------------------------------------------
    dataset = TensorDataset(
        X.float(),
        y.float()
    )

    train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    # ---------------------------------------------------------
    # 4. W-DRO 모델 생성
    # ---------------------------------------------------------
    wdro = WassersteinDRO(
        epsilon=0.1,
        lr=1e-3,
        inner_steps=10,
        inner_lr=0.02
    )

    print("Device:", wdro.device)

    # ---------------------------------------------------------
    # 5. W-DRO 학습
    # ---------------------------------------------------------
    wdro.train(
        train_loader,
        epochs=100
    )

    print("W-DRO training test finished!")

    # ---------------------------------------------------------
    # 6. evaluate() 테스트
    # ---------------------------------------------------------
    result = wdro.evaluate(
        train_loader
    )

    print(
        "Training MSE:",
        result["mse"]
    )

    # ---------------------------------------------------------
    # 7. predict() 테스트
    # ---------------------------------------------------------
    predictions = wdro.predict(
        train_loader
    )

    print(
        "Prediction shape:",
        predictions.shape
    )

    # ---------------------------------------------------------
    # 8. save() 테스트
    # ---------------------------------------------------------
    checkpoint_path = "wdro_test.pt"

    wdro.save(
        checkpoint_path
    )

    print(
        "Checkpoint saved:",
        checkpoint_path
    )

    # ---------------------------------------------------------
    # 9. load() 테스트
    # ---------------------------------------------------------
    loaded_wdro = WassersteinDRO()

    loaded_wdro.load(
        checkpoint_path
    )

    loaded_result = loaded_wdro.evaluate(
        train_loader
    )

    print(
        "Loaded model MSE:",
        loaded_result["mse"]
    )

    print("Common interface test passed!")