from dcgan import DCGAN
import numpy as np
from keras.optimizers import Adam
from data_loader import load_cucumber


def normalize(X):
    return (X-127.5)/127.5


def denormalize(X):
    return ((X + 1.0)/2.0*255.0).astype(dtype=np.uint8)


if __name__ == '__main__':
    batch_size = 16
    epochs = 1000
    input_dim = 30
    g_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
    d_optim = Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)

    ### 0. prepare data
    X_train, X_test, y_test = load_cucumber()
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    input_shape = X_train[0].shape
    X_test_original = X_test.copy()

    ### 1. train generator & discriminator
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.compile(g_optim, d_optim)
    g_losses, d_losses = dcgan.train(epochs, batch_size, X_train)
    with open('loss.csv', 'w') as f:
        for g_loss, d_loss in zip(g_losses, d_losses):
            f.write(str(g_loss) + ',' + str(d_loss) + '\n')