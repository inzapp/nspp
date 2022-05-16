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
    raw_batch_x = []
    raw_batch_y = []
    for i in range(len(data) - time_step - 1):
        batch, _, _ = transform(data[i:(i + time_step + 1)], time_step)
        batch_x.append(batch[:time_step])
        batch_y.append(batch[-1])
    batch_x = np.asarray(batch_x).reshape((len(batch_x), time_step, 1)).astype('float32')
    batch_y = np.asarray(batch_y).reshape((len(batch_y), 1, 1)).astype('float32')
    return batch_x, batch_y


def load_data():
    import yfinance as yf
    data = []
    data = yf.download('AMZN', start='2013-01-01', end='2022-05-01', progress=False)['Close'].values

    # with open('./time_series_covid_19_confirmed.csv') as f:
    #     for line in f.readlines():
    #         if line.find(r'"Korea, South"') > -1:
    #             data = list(map(float, line.split(',')[5:]))
    #             break

    # with open('./cansim-0800020-eng-6674700030567901031.csv') as f:
    #     for line in f.readlines():
    #         val = float(line.split(',')[1])
    #         data.append(val)
    return data


@tf.function
def graph_forward(model, x, training):
    return model(x, training=training)


@tf.function
def compute_gradient(model, optimizer, batch_x, y_true):
    with tf.GradientTape() as tape:
        y_pred = graph_forward(model, batch_x, True)
        abs_error = tf.abs(y_true - y_pred)
        loss = tf.square(abs_error)
        loss = tf.reduce_mean(loss, axis=0)
        mae = tf.reduce_mean(abs_error)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return mae


def normalize(arr):
    arr = np.asarray(arr, dtype=np.float32)
    min_val = np.min(arr)
    if min_val < 0.0:
        arr += np.abs(min_val)
    arr /= np.max(arr)
    return arr


def evaluate(model, train_data, validation_data, time_step, mode='train', show_plot=False, future_step=0):
    print()
    x = None
    forward_length = 0
    if mode == 'train':
        x = list(train_data[:time_step])
        y_seq = []
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
        y = graph_forward(model, np.asarray(x).reshape((1, time_step, 1)), False)
        y = np.asarray(y).reshape(-1)
        x = np.append(x[1:], y[-1])
        x = inverse_transform(x, criteria, max_val)
        y_seq.append(float(x[-1]))

    y_true = None
    y_pred = None
    if mode == 'train':
        y_true = np.asarray(train_data[time_step:])
        y_pred = np.asarray(y_seq)
    elif mode == 'validation':
        y_true = np.asarray(validation_data)
        y_pred = np.asarray(y_seq)

    loss = -1.0
    if future_step == 0:
        loss = np.mean(np.abs(np.asarray(y_true - y_pred)))
        print(f'validation loss : {loss:4f}\n')
    if show_plot:
        plot(y_true, y_pred)
    return loss


def fit(model, train_data, validation_data, train_x, train_y, validation_x, validation_y, batch_size, time_step, max_iteration_count):
    os.makedirs('checkpoints', exist_ok=True)
    optimizer = tf.keras.optimizers.RMSprop(lr=0.0075)
    # optimizer = tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    # optimizer = tf.keras.optimizers.Adam(lr=0.01, beta_1=0.5, beta_2=0.95)
    # optimizer = tf.keras.optimizers.Adam(lr=0.0075)
    iteration_count = 0
    print(f'\ntrain on {len(train_x)} samples. train_x.shape : {train_x.shape}')
    print(f'validate on {len(validation_x)} samples. validation_x.shape : {validation_x.shape} ')
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
            if iteration_count % 1000 == 0:
                loss = evaluate(model, train_data, validation_data, time_step, mode='validation')
                model.save(f'checkpoints/model_{iteration_count}_iter_{loss:.2f}_val_loss.h5', include_optimizer=False)
            if iteration_count == max_iteration_count:
                return


def main():
    data = load_data()
    # plt.plot(data)
    # plt.legend(['data'])
    # plt.show()
    # exit(0)

    time_step = 64
    batch_size = 64

    validation_ratio = 0.2
    split = int(len(data) * (1.0 - validation_ratio))
    train_data = data[:split]
    validation_data = data[split:]
    assert len(train_data) > time_step
    assert len(validation_data) > time_step

    train_x, train_y = make_time_series_data(train_data, time_step)
    validation_x, validation_y = make_time_series_data(validation_data, time_step)

    input_layer = tf.keras.layers.Input(shape=(time_step, 1))
    x = tf.keras.layers.LSTM(units=64, kernel_initializer='glorot_normal', activation='tanh', return_sequences=True)(input_layer)
    x = tf.keras.layers.LSTM(units=64, kernel_initializer='glorot_normal', activation='tanh', return_sequences=True)(x)
    x = tf.keras.layers.Dense(units=1, activation='linear')(x)
    model = tf.keras.models.Model(input_layer, x)
    # model = tf.keras.models.load_model('checkpoints/model_43000_iter_240.28371114527926_val_loss.h5', compile=False)
    model.summary()

    fit(model, train_data, validation_data, train_x, train_y, validation_x, validation_y, batch_size=batch_size, time_step=time_step, max_iteration_count=50000)
    print('\n')
    evaluate(model, train_data, validation_data, time_step, mode='train', show_plot=True)
    evaluate(model, train_data, validation_data, time_step, mode='validation', show_plot=True)
    evaluate(model, train_data, validation_data, time_step, mode='validation', future_step=100, show_plot=True)


if __name__ == '__main__':
    main()
