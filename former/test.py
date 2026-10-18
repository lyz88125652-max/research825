import torch
import torch.nn as nn

# --- 1. データ ---
x = torch.tensor([[1.0]])   # 例として x=1 の1点だけ使う
y = torch.tensor([[0.5]])   # 目的値

# --- 2. 2層NN（小さくする） ---


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)  # 1→2
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)  # 2→1


model = SimpleNN()

# SGD にする（更新式が単純になる）
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# --- 3. 1回目のepoch前の重みを保存 ---
W1_before = model.fc1.weight.data.clone()
b1_before = model.fc1.bias.data.clone()
W2_before = model.fc2.weight.data.clone()
b2_before = model.fc2.bias.data.clone()

# --- 4. forward ---
z1 = model.fc1(x)
a1 = model.relu(z1)
z2 = model.fc2(a1)
loss = criterion(z2, y)

print("=== Forward ===")
print(f"x = {x.item()}")
print(f"z1 = W1*x + b1 = {z1.detach().numpy()}")
print(f"a1 = ReLU(z1) = {a1.detach().numpy()}")
print(f"z2 = W2*a1 + b2 = {z2.detach().numpy()}")
print(f"loss = {loss.item()}")

# --- 5. backward ---
optimizer.zero_grad()
loss.backward()

grad_W1 = model.fc1.weight.grad.clone()
grad_b1 = model.fc1.bias.grad.clone()
grad_W2 = model.fc2.weight.grad.clone()
grad_b2 = model.fc2.bias.grad.clone()

print("\n=== Gradients ===")
print(f"dL/dW1 = {grad_W1.numpy()}")
print(f"dL/db1 = {grad_b1.numpy()}")
print(f"dL/dW2 = {grad_W2.numpy()}")
print(f"dL/db2 = {grad_b2.numpy()}")

# --- 6. update (SGD) ---
optimizer.step()

W1_after = model.fc1.weight.data.clone()
b1_after = model.fc1.bias.data.clone()
W2_after = model.fc2.weight.data.clone()
b2_after = model.fc2.bias.data.clone()

print("\n=== Update (SGD) ===")
print("W ← W - lr * dL/dW")
print(f"W1_before =\n{W1_before.numpy()}")
print(f"W1_after  =\n{W1_after.numpy()}")

print(f"\nb1_before = {b1_before.numpy()}")
print(f"b1_after  = {b1_after.numpy()}")

print(f"\nW2_before =\n{W2_before.numpy()}")
print(f"W2_after  =\n{W2_after.numpy()}")

print(f"\nb2_before = {b2_before.numpy()}")
print(f"b2_after  = {b2_after.numpy()}")
