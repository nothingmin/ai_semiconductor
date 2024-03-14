import numpy as np
import matplotlib.pyplot as plt

# given
m_train = 1000
w = 2
b = 3
x_data = 10 * np.array(range(m_train)) / m_train
y_data = x_data * w + np.random.randn(m_train) + b


def forward(x, w, b):
    return x * w + b


def loss(x, y, w, b):
    y_pred = forward(x, w, b)
    return (y_pred - y) ** 2


def gradient_w(x, y, w, b):
    return 2 * x * (x * w + b - y)


def gradient_b(x, y, w, b):
    return 2 * (x * w + b - y)


# train
w_pred = 1.0
b_pred = 1.0
lr = 0.01
epoch = 1000
plt.plot(x_data, y_data)
plt.ylabel('y')
plt.xlabel('x')
plt.show()
for i in range(epoch):
    gr_w = 0
    gr_b = 0
    cost = 0
    for x, y in zip(x_data, y_data):
        gr_w += gradient_w(x, y, w_pred, b_pred)
        gr_b += gradient_b(x, y, w_pred, b_pred)
        cost += loss(x, y, w_pred, b_pred)
    w_pred = w_pred - lr * gr_w / m_train
    b_pred = b_pred - lr * gr_b / m_train
    cost = cost / m_train
    if i % 2 == 0:
        print("epoch=%d, cost=%f,w'=%f b'=%f" % (i, cost, w_pred, b_pred))
