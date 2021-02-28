import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot(y1, y2):
    plt.plot(y1)
    plt.plot(y2)
    plt.legend(['train_x', 'prediction'])
    plt.show()


def main():
    train_data = []
    with open(r'C:\inz\train_data\covid\time_series_covid_19_confirmed.csv') as f:
        for line in f.readlines():
            if line.find(r'"Korea, South"') > -1:
                train_data = list(map(int, line.split(',')[5:]))
                break

    # normalize
    train_data = np.asarray(train_data)
    max_val = np.max(train_data)
    min_val = np.min(train_data)
    train_data = (train_data - min_val) / max_val

    train_x = np.asarray(train_data[:-1])
    train_y = np.asarray(train_data[1:])
    train_x = train_x.reshape((len(train_x), 1, 1)).astype('float32')
    train_y = train_y.reshape((len(train_y), 1, 1)).astype('float32')

    input_layer = tf.keras.layers.Input(shape=train_x.shape[1:])
    x = tf.keras.layers.LSTM(units=128, return_sequences=True)(input_layer)
    x = tf.keras.layers.LSTM(units=128, return_sequences=True)(x)
    x = tf.keras.layers.Dense(units=1)(x)
    model = tf.keras.models.Model(input_layer, x)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-2),
        loss=tf.keras.losses.MeanSquaredError())

    model.fit(
        x=train_x,
        y=train_y,
        epochs=100,
        batch_size=128)

    y_pred = []
    y = model.predict(x=train_x.reshape(-1)[0].reshape(1, 1), batch_size=1)
    y_pred.append(y.reshape(-1)[0])
    for _ in tqdm(range(362)):
        y = model.predict(x=y.reshape(1, 1), batch_size=1)
        y_pred.append(y.reshape(-1)[0])

    plot(train_x.reshape(-1), y_pred)


if __name__ == '__main__':
    main()
