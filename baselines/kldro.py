import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common.mlp import build_mlp
from common.scm import forward_scm
from experiments.test_kldro_inner import worst_case_weights


class KLDRO:
    def __init__(self, eps=0.1, lr=1e-3):
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")

        self.model = build_mlp()
        self.eps = eps

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        self.criterion = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr
        )

    def train(self, train_loader, epochs=100):
        self.model.train()

        for epoch in range(epochs):
            total_epoch_loss = 0.0
            total_samples = 0

            for x, y in train_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)
                raw_losses = self.criterion(
                    predictions,
                    y
                )
                if raw_losses.ndim > 1:
                    sample_losses = raw_losses.flatten(
                        start_dim=1
                    ).mean(dim=1)
                else:
                    sample_losses = raw_losses

                weights = worst_case_weights(
                    sample_losses.detach(),
                    self.eps
                ).to(
                    device=sample_losses.device,
                    dtype=sample_losses.dtype
                )

                robust_loss = torch.sum(
                    weights * sample_losses
                )

                self.optimizer.zero_grad()
                robust_loss.backward()
                self.optimizer.step()

                total_epoch_loss += robust_loss.item() * x.size(0)
                total_samples += x.size(0)

            epoch_loss = total_epoch_loss / total_samples

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] "
                    f"Loss: {epoch_loss:.6f}"
                )

    def predict(self, data_loader):
        self.model.eval()
        all_predictions = []

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                all_predictions.append(
                    self.model(x).cpu()
                )

        return torch.cat(all_predictions, dim=0)

    def evaluate(self, data_loader):
        self.model.eval()
        total_squared_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)
                total_squared_error += (
                    ((predictions - y) ** 2).sum().item()
                )
                total_samples += y.numel()

        return {
            "mse": total_squared_error / total_samples
        }

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "eps": self.eps,
            },
            path
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=True
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        self.eps = checkpoint["eps"]
        self.model.to(self.device)


if __name__ == "__main__":
    torch.manual_seed(0)

    n_samples = 128
    exogenous = [torch.randn(n_samples) for _ in range(5)]
    X, y = forward_scm(*exogenous)

    dataset = TensorDataset(
        X.float(),
        y.float()
    )
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    kldro = KLDRO(
        eps=0.1,
        lr=1e-3
    )
    print("Device:", kldro.device)

    kldro.train(
        train_loader,
        epochs=2
    )

    result = kldro.evaluate(train_loader)
    print("Training MSE:", result["mse"])

    predictions = kldro.predict(train_loader)
    print("Prediction shape:", predictions.shape)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as checkpoint:
        checkpoint_path = checkpoint.name

    try:
        kldro.save(checkpoint_path)
        print("Checkpoint saved:", checkpoint_path)

        loaded_kldro = KLDRO(eps=0.0)
        loaded_kldro.load(checkpoint_path)
        loaded_result = loaded_kldro.evaluate(train_loader)
        print("Loaded model MSE:", loaded_result["mse"])

        assert predictions.shape == (n_samples, 1)
        assert loaded_kldro.eps == kldro.eps
        assert abs(loaded_result["mse"] - result["mse"]) < 1e-6
        print("Common interface test passed!")
    finally:
        os.remove(checkpoint_path)
