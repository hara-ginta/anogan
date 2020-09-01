import os, cv2
from dcgan import DCGAN
from anogan import ANOGAN
import numpy as np
from keras.optimizers import Adam
from data_loader import load_cucumber
from train import normalize, denormalize


if __name__ == '__main__':
    iterations = 100
    input_dim = 30
    anogan_optim = Adam(lr=0.001, amsgrad=True)

    ### 0. prepare data
    X_train, X_test, y_test = load_cucumber()
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    input_shape = X_train[0].shape

    ### 1. train generator & discriminator
    dcgan = DCGAN(input_dim, input_shape)
    dcgan.load_weights('weights/generator_3999.h5', 'weights/discriminator_3999.h5')

    for i, test_img in enumerate(X_test):
        test_img = test_img[np.newaxis,:,:,:]
        anogan = ANOGAN(input_dim, dcgan.g)
        anogan.compile(anogan_optim)
        anomaly_score, generated_img = anogan.compute_anomaly_score(test_img, iterations)
        generated_img = denormalize(generated_img)
        imgs = np.concatenate((denormalize(test_img[0]), generated_img[0]), axis=1)
        cv2.imwrite('predict' + os.sep + str(int(anomaly_score)) + '_' + str(i) + '.png', imgs)
        print(str(i) + ' %.2f'%anomaly_score)
        with open('scores.txt', 'a') as f:
            f.write(str(anomaly_score) + '\n')