import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm


class MeanAbsoluteLogError(tf.keras.losses.Loss):
    """
    Mean absolute logarithmic error loss function.
    f(x) = -log(1 - MAE(x))
    Usage:
        model.compile(loss=[MeanAbsoluteLogError()], optimizer="sgd")
        """

    def call(self, y_true, y_pred):
        from tensorflow.python.framework.ops import convert_to_tensor_v2
        y_pred = convert_to_tensor_v2(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss = -tf.math.log(1.0 + 1e-7 - tf.math.abs(y_pred - y_true))
        return tf.keras.backend.mean(loss)


def plot(y_true, y_pred):
    plt.plot(y_true)
    # for i in range(len(y_true)):
    #     y_pred[i] = None
    plt.plot(y_pred)
    plt.legend(['train data', 'predicted data'])
    plt.show()


def normalize(arr):
    arr = np.asarray(arr)
    max_val = np.max(arr)
    min_val = np.min(arr)
    arr -= min_val
    arr /= max_val
    return arr, min_val, max_val


def denormalize(arr, min_val, max_val):
    arr = np.asarray(arr)
    arr *= max_val
    arr += min_val
    return arr


def make_time_series_data(data, time_step):
    batch_x = []
    batch_y = []
    for i in range(len(data) - time_step - 1):
        batch, _, _ = normalize(data[i:(i + time_step + 1)])
        batch_x.append(batch[:time_step])
        batch_y.append(batch[-1])
    batch_x = np.asarray(batch_x).reshape((len(batch_x), time_step, 1)).astype('float32')
    batch_y = np.asarray(batch_y).reshape((len(batch_y), 1, 1)).astype('float32')
    return batch_x, batch_y


def predict_and_plot_data(model, data, time_step):
    x = data[0].to_list()
    y_pred = train_x[0].to_list()
    for _ in tqdm(range(len(data) - time_step)):
        y = model(np.asarray(x).reshape((1, time_step, 1)), training=False)
        y = np.asarray(y).reshape(-1)
        y_pred.append(y[-1])
        x.append(y[-1])
        x.pop(0)
    plot(data, y_pred)


def main():
    data = []
    with open('./time_series_covid_19_confirmed.csv') as f:
        for line in f.readlines():
            if line.find(r'"Korea, South"') > -1:
                data = list(map(int, line.split(',')[5:]))
                break

    time_step = 64
    batch_size = 1

    split = int(len(data) * validation_ratio)
    train_data = data[:split]
    validation_data = data[split:]
    train_x, train_y = make_time_series_data(train_data)
    validation_x, validation_y = make_time_series_data(validation_data)

    print(train_x.shape)
    print(train_y.shape)
    print(validation_x.shape)
    print(validation_y.shape)

    input_layer = tf.keras.layers.Input(shape=(time_step, 1))
    x = tf.keras.layers.LSTM(units=16, return_sequences=True)(input_layer)
    # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='linear'))(x)
    x = tf.keras.layers.Dense(units=1, activation='linear')(x)
    model = tf.keras.models.Model(input_layer, x)
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001), loss=tf.keras.losses.MeanSquaredError())
    model.fit(
        x=train_x,
        y=train_y,
        epochs=20,
        batch_size=batch_size)



if __name__ == '__main__':
    main()
