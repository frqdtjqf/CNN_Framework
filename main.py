import numpy as np
from src.Layer import convolution, pooling, activation, fully_connected, softmax, flatten, base


# --------------------------
# Dummy Input & Labels
# --------------------------
batch_size = 2
num_classes = 10
X = np.random.rand(batch_size, 3, 32, 32)       # batch=2, 3 channels, 32x32 images
Y = np.zeros((batch_size, num_classes))
for i in range(batch_size):
    Y[i, np.random.randint(0, num_classes)] = 1  # zufällige One-Hot Labels

# --------------------------
# Build Model
# --------------------------
model: list[base.Layer] = [
    convolution.Convolution(filter_size=3, num_filters=8, in_channels=3),
    activation.Activation(),
    pooling.Pooling(pool_size=2),
    flatten.Flatten(),
    fully_connected.FullyConnected(num_input_neurons=8*15*15, num_neurons=num_classes)
]

# --------------------------
# Training Loop (Dummy)
# --------------------------
lr = 0.01
loss_fn = softmax.SoftmaxCrossEntropyLoss()

for step in range(1000):
    # Forward
    out = X
    for layer in model:
        out = layer.forward(out)

    # Loss
    loss = loss_fn.forward(out, Y)
    print(f"Step {step}, Loss: {loss:.5f}")

    # Backward
    grad = loss_fn.backward()
    for layer in reversed(model):
        grad = layer.backward(grad)

    # Update
    for layer in model:
        layer.update_params(lr)