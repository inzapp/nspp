import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot(y1, y2):
    plt.plot(y1)
    # for i in range(len(y1)):
    #     y2[i] = None
    plt.plot(y2)
    plt.legend(['train_x', 'prediction'])
    plt.show()


def make_time_series_data(train_data, time_step):
    time_step += 1
    ts_train_x = []
    ts_train_y = []
    for i in range(len(train_data) - time_step):
        time_step_data = train_data[i:i + time_step]
        ts_train_x.append(time_step_data[:-1])
        ts_train_y.append(time_step_data[-1])
    ts_train_x = np.asarray(ts_train_x)
    ts_train_y = np.asarray(ts_train_y)
    ts_train_x = ts_train_x.reshape(ts_train_x.shape + (1,)).astype('float32')
    ts_train_y = ts_train_y.reshape(ts_train_y.shape + (1,)).astype('float32')
    return ts_train_x, ts_train_y


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

    time_step = 128
    batch_size = 32

    train_x, train_y = make_time_series_data(train_data, time_step)
    print(train_x.shape)
    print(train_y.shape)

    input_layer = tf.keras.layers.Input(shape=train_x.shape[1:])
    x = tf.keras.layers.LSTM(units=32, return_sequences=True)(input_layer)
    x = tf.keras.layers.LSTM(units=32, return_sequences=True)(x)
    x = tf.keras.layers.Dense(units=1, kernel_initializer='he_uniform', activation='linear')(x)
    model = tf.keras.models.Model(input_layer, x)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-2),
        loss=tf.keras.losses.MeanSquaredError())

    model.fit(
        x=train_x,
        y=train_y,
        epochs=500,
        batch_size=batch_size)

    input_x = list(train_x.reshape(-1)[:time_step])
    y_pred = list(train_x.reshape(-1)[:time_step])
    for _ in tqdm(range(len(train_data) - time_step + len(train_data))):
        y = model.predict(x=np.asarray(input_x).reshape((1, time_step, 1)), batch_size=batch_size)
        y = y.reshape(-1)
        y_pred.append(y[-1])
        input_x = input_x[1:] + [float(y[-1])]
    plot(train_data, y_pred)


if __name__ == '__main__':
    main()
