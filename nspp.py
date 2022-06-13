import os 
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm
import yfinance as yf


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


class NasdaqStockPricePredictor:
    def __init__(self,
                 ticker,
                 start_date,
                 end_date,
                 interval='1d',
                 lr=0.001,
                 time_step=14,
                 batch_size=32,
                 future_step=7,
                 validation_ratio=0.2,
                 max_iteration_count=100000,
                 pretrained_model_path=''):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.time_step = time_step
        self.batch_size = batch_size
        self.lr = lr
        self.future_step = future_step
        self.validation_ratio = validation_ratio
        self.max_iteration_count = max_iteration_count
        self.model = None
        if pretrained_model_path != '':
            self.model = self.load_model(pretrained_model_path)
        else:
            self.model = self.build_model()
        self.train_data, self.validation_data, self.train_x, self.train_y, self.validation_x, self.validation_y = self.load_data()

    def load_model(self, model_path):
        model_path = model_path.replace('\\', '/')
        model = tf.keras.models.load_model(model_path, compile=False)
        sp = model_path.split('/')[-1].split('_')
        self.ticker = sp[0]
        self.start_date = sp[1]
        self.end_date = sp[2]
        self.interval = sp[3]
        self.validation_ratio = float(sp[4])
        print(f'\nload pretrained model : [{model_path}]')
        print(f'ticker : {self.ticker}')
        print(f'start_date : {self.start_date}')
        print(f'end_date : {self.end_date}')
        print(f'interval : {self.interval}')
        print(f'validation_ratio : {self.validation_ratio}')
        return model

    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.time_step, 1))
        x = tf.keras.layers.LSTM(units=32, kernel_initializer='glorot_normal', activation='tanh', return_sequences=True)(input_layer)
        x = tf.keras.layers.LSTM(units=32, kernel_initializer='glorot_normal', activation='tanh', return_sequences=True)(x)
        x = tf.keras.layers.Dense(units=1, kernel_initializer='he_normal', activation='linear')(x)
        return tf.keras.models.Model(input_layer, x)

    def load_data(self):
        data = []
        if os.path.exists(self.ticker) and os.path.isfile(self.ticker):
            with open(self.ticker, 'rt') as f:
                lines = f.readlines()
            for line in lines:
                data.append(float(line))
            data = np.array(data)
        else:
            assert self.interval in ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
            if self.start_date < '1971-01-01':
                self.start_date = '1971-01-01'
            try:
                data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.interval, auto_adjust=True, prepost=False, progress=False)['Close'].values
            except:
                print(f'invalid date range to get {self.ticker} data : [{self.start_date} ~ {self.end_date}]')
                exit(0)
            nan_indexes = np.where(np.isnan(data) == True)
            if len(nan_indexes) > 0:
                ans = input(f'{len(nan_indexes)} nan detected in {self.ticker}({self.start_date} ~ {self.end_date}) data. ignore and continue train? [y/n] : ')
                if ans == 'y' or ans == 'Y':
                    for i in nan_indexes[::-1]:
                        data = np.delete(data, i)
                else:
                    exit(0)
        split = int(len(data) * (1.0 - self.validation_ratio))
        train_data = data[:split]
        validation_data = data[split:]
        assert len(train_data) > self.time_step 
        if self.validation_ratio > 0.0:
            assert len(validation_data) > self.time_step
        train_x, train_y = self.convert_time_series_data(train_data, self.time_step)
        validation_x, validation_y = self.convert_time_series_data(validation_data, self.time_step)
        return train_data, validation_data, train_x, train_y, validation_x, validation_y

    def plot_train_validation_data(self, train_data, validation_data):
        len_train = len(train_data)
        len_validation = len(validation_data)
        len_sum = len_train + len_validation
        padded_train_data = train_data.tolist() + [None for _ in range(len_sum - len_train)]
        padded_validation_data = [None for _ in range(len_sum - len_validation - 1)] + [train_data[-1]] + validation_data.tolist()
        self.plot([padded_train_data, padded_validation_data], [f'{self.ticker} train', f'{self.ticker} validation'], f'{self.ticker} train data')

    def convert_time_series_data(self, data, time_step):
        data_x = []
        data_y = []
        for i in range(len(data) - time_step - 1):
            batch, _, _ = self.transform(data[i:(i + time_step + 1)])
            data_x.append(batch[:time_step])
            data_y.append(batch[-1])
        data_x = np.asarray(data_x).reshape((len(data_x), time_step, 1)).astype('float32')
        data_y = np.asarray(data_y).reshape((len(data_y), 1, 1)).astype('float32')
        return data_x, data_y

    def transform(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        max_val = np.max(np.abs(arr[:self.time_step])) + 1e-5
        arr /= max_val
        criteria = arr[0]
        arr -= criteria
        return arr, criteria, max_val

    def inverse_transform(self, arr, criteria, max_val):
        arr = np.asarray(arr, dtype=np.float32)
        arr += criteria
        arr *= max_val
        return arr

    def plot(self, datas, legend, title):
        plt.figure(figsize=(13, 8))
        plt.gcf().canvas.set_window_title(title)
        for data in datas:
            plt.plot(data)
        plt.legend(legend)
        plt.tight_layout(pad=0.5)
        plt.show()

    @tf.function
    def graph_forward(self, model, x, training):
        return model(x, training=training)

    @tf.function
    def compute_gradient(self, model, optimizer, batch_x, y_true):
        with tf.GradientTape() as tape:
            y_pred = self.graph_forward(model, batch_x, True)
            abs_error = tf.abs(y_true - y_pred)
            loss = tf.square(abs_error)
            loss = tf.reduce_mean(loss, axis=0)
            mae = tf.reduce_mean(abs_error)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return mae

    def shuffle_data(self, x, y):
        assert len(x) == len(y)
        r = np.arange(len(x))
        np.random.shuffle(r)
        x = x[r]
        y = y[r]
        return x, y

    def fit(self):
        print(f'\ntrain on {len(self.train_x)} samples. train_x.shape : {self.train_x.shape}')
        print(f'validate on {len(self.validation_x)} samples. validation_x.shape : {self.validation_x.shape} ')
        self.plot_train_validation_data(self.train_data, self.validation_data)
        self.model.summary()
        os.makedirs('checkpoints', exist_ok=True)
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        iteration_count = 0
        break_flag = False
        while True:
            self.train_x, self.train_y = self.shuffle_data(self.train_x, self.train_y)
            batch_index = 0
            while True:
                start_index = batch_index * self.batch_size
                end_index = start_index + self.batch_size
                if start_index > len(self.train_x) or end_index > len(self.train_x):
                    break
                batch_x = self.train_x[start_index:end_index]
                batch_y = self.train_y[start_index:end_index]
                loss = self.compute_gradient(self.model, optimizer, batch_x, batch_y)
                iteration_count += 1
                batch_index += 1
                print(f'\r[{iteration_count:6d} iter] loss => {loss:.4f}', end='')
                if iteration_count % 2000 == 0:
                    model_save_path = f'checkpoints/{self.ticker}_{iteration_count}_iter.h5'
                    if self.validation_ratio > 0.0:
                        mape, mae = self.evaluate_validation()
                        model_save_path = f'checkpoints/{self.ticker}_{self.start_date}_{self.end_date}_{self.interval}_{self.validation_ratio}_val_{iteration_count}_iter_{mape:.4f}_MAPE_{mae:.4f}_MAE.h5'
                    else:
                        print()
                    self.model.save(model_save_path, include_optimizer=False)
                if iteration_count == self.max_iteration_count:
                    break_flag = True
                    break
            if break_flag:
                break
        self.evaluate_train(show_plot=True, future_step=self.future_step)
        if self.validation_ratio > 0.0:
            self.evaluate_validation(show_plot=True, future_step=self.future_step)

    def evaluate_train(self, show_plot=False, future_step=0):
        print()
        y_pred_initial_x = self.train_data[:self.time_step]
        y_pred_data_x = self.train_data[self.time_step:]
        y_pred = self.predict(y_pred_initial_x, y_pred_data_x, 0)
        future_pred_initial_x = self.train_data[-self.time_step:]
        future_pred = self.predict(future_pred_initial_x, [], future_step)
        y_true = np.asarray(y_pred_data_x)
        y_pred = np.asarray(y_pred)
        mae = np.mean(np.abs(np.asarray(y_true - y_pred)))
        mape = np.mean(mae / np.abs(y_true))
        print(f'train MAPE : {mape:.4f}, MAE : {mae:4f}\n')
        if show_plot:
            if future_step > 0:
                self.plot_with_future_prediction(y_true, y_pred, future_pred, 'AI predicted train data with future')
            else:
                self.plot([y_true, y_pred], [f'{self.ticker}', 'AI predicted'], 'AI predicted train data')
        return mape, mae

    def evaluate_validation(self, show_plot=False, future_step=0):
        print()
        y_pred_initial_x = self.train_data[-self.time_step:]
        y_pred_data_x = self.validation_data
        y_pred = self.predict(y_pred_initial_x, y_pred_data_x, 0)
        future_pred_initial_x = self.validation_data[-self.time_step:]
        future_pred = self.predict(future_pred_initial_x, [], future_step)
        y_true = np.asarray(y_pred_data_x)
        y_pred = np.asarray(y_pred)
        mae = np.mean(np.abs(np.asarray(y_true - y_pred)))
        mape = np.mean(mae / np.abs(y_true))
        print(f'validation MAPE : {mape:.4f}, MAE : {mae:4f}\n')
        if show_plot:
            if future_step > 0:
                self.plot_with_future_prediction(y_true, y_pred, future_pred, 'AI predicted validation data with future', show_end_date=True)
            else:
                self.plot([y_true, y_pred], [f'{self.ticker}', 'AI predicted'], 'AI predicted validation data')
        return mape, mae

    def predict(self, initial_x, data_x, future_step=0):
        y_pred = []
        x = initial_x
        for i in tqdm(range(len(data_x) + future_step)):
            x, criteria, max_val = self.transform(x)
            y = self.graph_forward(self.model, np.asarray(x).reshape((1, self.time_step, 1)), False)
            x = np.append(x[1:], np.asarray(y).reshape(-1)[-1])
            x = self.inverse_transform(x, criteria, max_val)
            y_pred.append(float(x[-1]))
            if i < len(data_x):
                x = np.append(x[:-1], data_x[i])
        return y_pred

    def plot_with_future_prediction(self, y_true, y_pred, future_pred, title, show_end_date=False):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        future_pred_not_padded = np.array(future_pred)
        future_pred = [None for _ in range(len(y_true) - 1)] + [y_true[-1]] + future_pred
        future_pred = np.asarray(future_pred)
        if show_end_date:
            print(f'last price({self.end_date}) : {y_true[-1]:.4f}')
        else:
            print(f'last price : {y_true[-1]:.4f}')
        interval_n, interval_unit = self.get_interval_unit(self.interval)
        for i in range(len(future_pred_not_padded)):
            interval_n_cur = (i + 1) * interval_n
            if interval_n_cur > 1:
                print(f'predicted price {interval_n_cur:3d} {interval_unit}s after: {future_pred_not_padded[i]:.4f}')
            else:
                print(f'predicted price {interval_n_cur:3d}  {interval_unit} after: {future_pred_not_padded[i]:.4f}')
        self.plot([y_true, y_pred, future_pred], [f'{self.ticker}', 'AI predicted', 'AI predicted future'], title)

    def get_interval_unit(self, interval):
        n = ''
        unit = ''
        for i in range(len(interval)):
            if 48 <= ord(self.interval[i]) <= 57:
                n += self.interval[i]
            else:
                unit = self.interval[i:]
                break
        if unit == 'm':
            unit = 'minute'
        elif unit == 'h':
            unit = 'hour'
        elif unit == 'd':
            unit = 'day'
        elif unit == 'wk':
            unit = 'week'
        elif unit == 'mo':
            unit = 'month'
        return int(n), unit
