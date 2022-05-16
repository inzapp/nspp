import os 
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def plot(y_true, y_pred):
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.legend(['Real data', 'AI predicted'])
    plt.show()


def transform(arr, time_step):
    arr = np.asarray(arr, dtype=np.float32)
    max_val = np.max(arr[:time_step])
    arr /= np.max(arr)
    criteria = arr[0]
    arr -= criteria
    return arr, criteria, max_val


def inverse_transform(arr, criteria, max_val):
    arr = np.asarray(arr, dtype=np.float32)
    arr += criteria
    arr *= max_val
    return arr


def make_time_series_data(data, time_step):
    batch_x = []
    batch_y = []
    for i in range(len(data) - time_step - 1):
        batch, _, _ = transform(data[i:(i + time_step + 1)], time_step)
        batch_x.append(batch[:time_step])
        batch_y.append(batch[-1])
    batch_x = np.asarray(batch_x).reshape((len(batch_x), time_step, 1)).astype('float32')
    batch_y = np.asarray(batch_y).reshape((len(batch_y), 1, 1)).astype('float32')
    return batch_x, batch_y


def predict_and_plot_data(model, train_data, validation_data, time_step, mode='train', future_step=0):
    x = None
    forward_length = 0
    if mode == 'train':
        x = list(train_data[:time_step])
        y_seq = train_data[:time_step]
        forward_length = len(train_data) - time_step + future_step
    elif mode == 'validation':
        x = list(train_data[-time_step:])
        y_seq = []
        forward_length = len(validation_data) + future_step
    else:
        print(f'unknown mode {mode}')
        return
    for _ in tqdm(range(forward_length)):
        x, criteria, max_val = transform(x, time_step)
        y = model(np.asarray(x).reshape((1, time_step, 1)), training=False)
        y = np.asarray(y).reshape(-1)
        x = np.append(x[1:], y[-1])
        x = inverse_transform(x, criteria, max_val)
        y_seq.append(float(x[-1]))
    if mode == 'train':
        plot(train_data, y_seq)
    elif mode == 'validation':
        plot(validation_data, y_seq)


def load_data():
    data = []
    # with open('./time_series_covid_19_confirmed.csv') as f:
    #     for line in f.readlines():
    #         if line.find(r'"Korea, South"') > -1:
    #             data = list(map(int, line.split(',')[5:]))
    #             break

    with open('./cansim-0800020-eng-6674700030567901031.csv') as f:
        for line in f.readlines():
            val = float(line.split(',')[1])
            data.append(val)
    return data


@tf.function
def compute_gradient(model, optimizer, batch_x, y_true):
    with tf.GradientTape() as tape:
        y_pred = model(batch_x, training=False)
        loss = tf.square(y_true - y_pred)
        mse = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return mse


def fit(model, train_x, train_y, batch_size, max_iteration_count):
    os.makedirs('checkpoints', exist_ok=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.01)
    iteration_count = 0
    while True:
        r = np.arange(len(train_x))
        np.random.shuffle(r)
        train_x = train_x[r]
        train_y = train_y[r]
        batch_index = 0
        while True:
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            if start_index > len(train_x) or end_index > len(train_x):
                break
            batch_x = train_x[start_index:end_index]
            batch_y = train_y[start_index:end_index]
            loss = compute_gradient(model, optimizer, batch_x, batch_y)
            iteration_count += 1
            batch_index += 1
            print(f'\r[{iteration_count:6d} iter] loss => {loss:.4f}', end='')
            if iteration_count % 2000 == 0:
                model.save(f'checkpoints/model_{iteration_count}_iter.h5', include_optimizer=False)
                print()
            if iteration_count == max_iteration_count:
                return


def main():
    data = load_data()
    # plt.plot(data)
    # plt.legend(['data'])
    # plt.show()
    # exit(0)

    time_step = 50
    batch_size = 8

    validation_ratio = 0.2
    split = int(len(data) * (1.0 - validation_ratio))
    train_data = data[:split]
    validation_data = data[split:]
    assert len(train_data) > time_step
    assert len(validation_data) > time_step

    train_x, train_y = make_time_series_data(train_data, time_step)
    validation_x, validation_y = make_time_series_data(validation_data, time_step)

    input_layer = tf.keras.layers.Input(shape=(time_step, 1))
    x = tf.keras.layers.LSTM(units=32, return_sequences=True)(input_layer)
    x = tf.keras.layers.LSTM(units=32, return_sequences=True)(x)
    # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, activation='linear'))(x)
    x = tf.keras.layers.Dense(units=1, activation='linear')(x)
    model = tf.keras.models.Model(input_layer, x)
    model.summary()

    fit(model, train_x, train_y, batch_size=batch_size, max_iteration_count=2000)
    predict_and_plot_data(model, train_data, validation_data, time_step, mode='train')
    predict_and_plot_data(model, train_data, validation_data, time_step, mode='validation')
    predict_and_plot_data(model, train_data, validation_data, time_step, mode='validation', future_step=100)


def test():
    a = np.array([1, -1, 5, -5, 10, -10, 15], dtype=np.float32)
    for v in a:
        print(f'{v:.2f}', end=' ')
    print()

    t, c, m = transform(a, len(a))
    for v in t:
        print(f'{v:.2f}', end=' ')
    print()

    a = inverse_transform(t, c, m)
    for v in a:
        print(f'{v:.2f}', end=' ')
    print()


if __name__ == '__main__':
    # test()
    main()
