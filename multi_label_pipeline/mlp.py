"""Shared dense hidden layers followed by three independent sigmoid outputs."""


def build_network():
    from torch import nn
    # Flatten also accepts the (N, 1, 1280) input used by the shared CNN trainer.
    return nn.Sequential(
        nn.Flatten(), nn.Linear(1280, 256), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 3))


def train_predict(X_train, y_train, X_test, groups, output, args, seed):
    from .cnn import train_predict as train_neural
    return train_neural(X_train, y_train, X_test, groups, output, args, seed,
                        network_factory=build_network, model_name="MLP",
                        epochs=args.mlp_epochs, batch_size=args.mlp_batch_size)
