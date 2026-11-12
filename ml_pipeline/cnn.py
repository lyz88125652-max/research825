"""1D feature CNN. Its axis indexes embedding coordinates, not peptide residues."""

import csv
import random

import numpy as np


def seed_everything(torch, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_network():
    from torch import nn

    return nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.GELU(), nn.MaxPool1d(4),
        nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.GELU(),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(),
        nn.Linear(64 * 16, 128), nn.GELU(), nn.Dropout(0.3), nn.Linear(128, 3),
    )


def choose_device(torch, requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable.")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable.")
    return torch.device(requested)


def inner_split(y, groups, seed):
    from .data import make_splits

    count = min(5, min(len(np.unique(groups[y == c])) for c in range(3)))
    if count < 2:
        raise ValueError("Too few sequence groups for CNN internal validation.")
    return make_splits(y, groups, count, seed)[0]


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    import torch
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    seed_everything(torch, seed)
    device = choose_device(torch, args.device)

    def initialize(X, y):
        scaler = StandardScaler().fit(X)
        transformed = scaler.transform(X).astype(np.float32)
        dataset = TensorDataset(torch.from_numpy(transformed).unsqueeze(1),
                                torch.from_numpy(y).long())
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(dataset, batch_size=args.cnn_batch_size, shuffle=True,
                            generator=generator, num_workers=0)
        model = build_network().to(device)
        weights = len(y) / (3 * np.bincount(y, minlength=3))
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                     weight_decay=1e-4)
        return scaler, loader, model, criterion, optimizer

    def fit_epoch(loader, model, criterion, optimizer):
        model.train()
        total = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(X_batch), y_batch)
            if not torch.isfinite(loss).item():
                raise ValueError("CNN loss became non-finite.")
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(y_batch)
        return total / len(loader.dataset)

    def probabilities(model, scaler, X):
        model.eval()
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(X), args.cnn_batch_size):
                batch = scaler.transform(X[start:start + args.cnn_batch_size]).astype(np.float32)
                tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
                chunks.append(torch.softmax(model(tensor), dim=1).cpu().numpy())
        return np.concatenate(chunks)

    # Select epoch count using only the outer training fold, with separate groups.
    fit, valid = inner_split(y_train, groups, seed)
    scaler, loader, model, criterion, optimizer = initialize(X_train[fit], y_train[fit])
    best_loss, best_epoch, stale = float("inf"), 1, 0
    history = []
    for epoch in range(1, args.cnn_epochs + 1):
        train_loss = fit_epoch(loader, model, criterion, optimizer)
        probs = probabilities(model, scaler, X_train[valid])
        validation_loss = float(-np.log(np.clip(probs[np.arange(len(valid)),
                                                     y_train[valid]], 1e-15, 1)).mean())
        history.append(["selection", epoch, train_loss, validation_loss])
        print(f"    CNN epoch {epoch}: train_loss={train_loss:.4f} "
              f"validation_loss={validation_loss:.4f}", flush=True)
        if validation_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    # Refit scaler and a fresh CNN on the entire outer training fold.
    del model, optimizer
    seed_everything(torch, seed)
    scaler, loader, model, criterion, optimizer = initialize(X_train, y_train)
    for epoch in range(1, best_epoch + 1):
        loss = fit_epoch(loader, model, criterion, optimizer)
        history.append(["refit", epoch, loss, ""])
    probs = probabilities(model, scaler, X_test)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler_mean": torch.tensor(scaler.mean_),
        "scaler_scale": torch.tensor(scaler.scale_),
        "labels": [1, 2, 3], "input_dim": 1280, "epochs": best_epoch,
    }, output / "model.pt")
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase", "epoch", "train_loss", "validation_loss"])
        writer.writerows(history)
    return probs, {"selected_epochs": best_epoch, "device": str(device),
                   "inner_train_rows": len(fit), "inner_validation_rows": len(valid)}
