"""Three-logit feature CNN with BCE training and independent sigmoid outputs."""

import csv
import random
import numpy as np

from .data import make_holdout
from .metrics import binary_log_loss


def build_network():
    from torch import nn
    return nn.Sequential(
        nn.Conv1d(1, 32, 7, padding=3), nn.GELU(), nn.MaxPool1d(4),
        nn.Conv1d(32, 64, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(16),
        nn.Flatten(), nn.Linear(1024, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3))


def choose_device(torch, requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable.")
    return torch.device(requested)


def train_predict(X_train, y_train, X_test, groups, output, args, seed, *,
                  network_factory=None, model_name="CNN", epochs=None, batch_size=None):
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    device = choose_device(torch, args.device)
    network_factory = network_factory or build_network
    epochs = args.cnn_epochs if epochs is None else epochs
    batch_size = args.cnn_batch_size if batch_size is None else batch_size
    def reseed():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    def initialize(X, y):
        scaler = StandardScaler().fit(X)
        dataset = TensorDataset(torch.from_numpy(scaler.transform(X).astype(np.float32)).unsqueeze(1),
                                torch.from_numpy(y.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                            generator=torch.Generator().manual_seed(seed))
        model = network_factory().to(device)
        positives = y.sum(axis=0)
        if np.any(positives == 0) or np.any(positives == len(y)):
            raise ValueError(f"{model_name} training requires positive and negative examples for every label.")
        pos_weight = torch.tensor((len(y) - positives) / positives, dtype=torch.float32, device=device)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        return scaler, loader, model, loss, optimizer
    def fit_epoch(loader, model, criterion, optimizer):
        model.train(); total = 0.0
        for X, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X.to(device)), y.to(device))
            if not torch.isfinite(loss).item(): raise ValueError(f"Non-finite {model_name} loss.")
            loss.backward(); optimizer.step()
            total += float(loss.detach().cpu()) * len(y)
        return total / len(loader.dataset)
    def predict(model, scaler, X):
        model.eval(); result = []
        with torch.inference_mode():
            for i in range(0, len(X), batch_size):
                values = scaler.transform(X[i:i + batch_size]).astype(np.float32)
                logits = model(torch.from_numpy(values).unsqueeze(1).to(device))
                result.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(result)

    fit, valid = make_holdout(y_train, groups, seed, fraction=0.2)
    reseed()
    scaler, loader, model, criterion, optimizer = initialize(X_train[fit], y_train[fit])
    best_loss, best_epoch, stale = float("inf"), 1, 0
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = fit_epoch(loader, model, criterion, optimizer)
        train_eval = binary_log_loss(y_train[fit], predict(model, scaler, X_train[fit]))
        validation = binary_log_loss(y_train[valid], predict(model, scaler, X_train[valid]))
        history.append(["selection", epoch, train_loss, train_eval, validation])
        print(f"    {model_name} epoch {epoch}: train_eval={train_eval:.4f} validation={validation:.4f}", flush=True)
        if validation < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation, epoch, 0
        else: stale += 1
        if stale >= args.patience: break
    del model, optimizer
    reseed()
    scaler, loader, model, criterion, optimizer = initialize(X_train, y_train)
    for epoch in range(1, best_epoch + 1):
        loss = fit_epoch(loader, model, criterion, optimizer)
        history.append(["refit", epoch, loss, "", ""])
    p = predict(model, scaler, X_test)
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "epoch", "train_weighted_loss", "train_eval_loss", "validation_loss"])
        writer.writerows(history)
    return p, {"selected_epochs": best_epoch, "device": str(device),
               "inner_train_rows": len(fit), "inner_validation_rows": len(valid),
               "loss": "BCEWithLogitsLoss; training-only pos_weight"}
