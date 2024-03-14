x_data = [1, 2, 3]
y_data = [2, 4, 6]
m_train = len(x_data)


# our linear model
def forward_cal(w, x):
    return w * x


def loss_cal(w, x, y):
    y_pred = forward_cal(w, x)
    return (y_pred-y) ** 2


def gradient_cal(w, x, y):  # d_loss/d_w
    return 2 * x * (w * x - y)


# Training part
w = 1.0
lr = 0.03
# Training loop
for epoch in range(20):
    # initialize
    d_w = 0
    d_b = 0
    loss = 0
    for x_val, y_val in zip(x_data, y_data):
        d_w += gradient_cal(w, x_val, y_val)
        loss += loss_cal(w, x_val, y_val)
    loss = loss / m_train
    w = w - lr * d_w / m_train
    if epoch % 2 == 0:
        print("epoch=%d, previous_loss=%f,w'=%f" % (epoch, loss, w))
