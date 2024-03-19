import numpy as np
import matplotlib.pyplot as plt

m_train = 1000
w = 2
b = 3
x_data = 10 * np.array(range(m_train)) / m_train
y_data = x_data * w + np.random.randn(m_train) + b


def forward_cal(w, b, x):
    return x * w + b


def loss_cal(w, b, x, y, y_pred):
    return (y_pred - y) ** 2


def w_gradient(w, b, x, y, y_pred):
    return 2 * x * (y_pred - y)


def b_gradient(w, b, x, y, y_pred):
    return 2 * (y_pred - y)


w = 0
b = 0
lr = 0.02
Nepoch = 400
# Training loop
for epoch in range(Nepoch):
    # initialize
    d_w = 0
    d_b = 0
    loss = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred = forward_cal(w, b, x_val)
        d_w += w_gradient(w, b, x_val, y_val, y_pred)
        d_b += b_gradient(w, b, x_val, y_val, y_pred)
        loss += loss_cal(w, b, x_val, y_val, y_pred)
    loss = loss / m_train
    w = w - lr * d_w / m_train
    b = b - lr * d_b / m_train
    if epoch % 100 == 0:
        print("epoch=%d, previous_loss=%f, w'=%f, b'=%f" % (epoch, loss, w, b))
plt.figure()
plt.plot(x_data, y_data, '*', label='Training data')
plt.plot(x_data, w * x_data + b, label='Trained model')
plt.xlabel('Input x')
plt.ylabel('Output y')
plt.legend()
plt.show()
