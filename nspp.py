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
                 lr=0.01,
                 time_step=14,
                 batch_size=32,
                 future_step=7,
                 validation_ratio=0.2,
                 max_iteration_count=100000,
                 pretrained_model_path=''):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.time_step = time_step
        self.batch_size = batch_size
        self.lr = lr
        self.future_step = future_step
        self.validation_ratio = validation_ratio
        self.max_iteration_count = max_iteration_count
        self.model = None
        if pretrained_model_path != '':
            self.model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        else:
            self.model = self.build_model()
        self.train_data, self.validation_data, self.train_x, self.train_y, self.validation_x, self.validation_y = self.load_data()

    def build_model(self):
        input_layer = tf.keras.layers.Input(shape=(self.time_step, 1))
        x = tf.keras.layers.LSTM(units=32, kernel_initializer='glorot_normal', activation='tanh', return_sequences=True)(input_layer)
        x = tf.keras.layers.LSTM(units=32, kernel_initializer='glorot_normal', activation='tanh', return_sequences=True)(x)
        x = tf.keras.layers.Dense(units=1, kernel_initializer='he_normal', activation='linear')(x)
        return tf.keras.models.Model(input_layer, x)

    def load_data(self):
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, progress=False)['Close'].values
        # data = []
        # with open('./cansim-0800020-eng-6674700030567901031.csv') as f:
        #     for line in f.readlines():
        #         val = float(line.split(',')[1])
        #         data.append(val)
        # data = np.array(data)
        split = int(len(data) * (1.0 - self.validation_ratio))
        train_data = data[:split]
        validation_data = data[split:]
        self.plot_train_validation_data(train_data, validation_data)
        assert len(train_data) > self.time_step and len(validation_data) > self.time_step
        train_x, train_y = self.convert_time_series_data(train_data, self.time_step)
        validation_x, validation_y = self.convert_time_series_data(validation_data, self.time_step)
        return train_data, validation_data, train_x, train_y, validation_x, validation_y

    def plot_train_validation_data(self, train_data, validation_data):
        len_train = len(train_data)
        len_validation = len(validation_data)
        len_sum = len_train + len_validation
        padded_train_data = train_data.tolist() + [None for _ in range(len_sum - len_train)]
        padded_validation_data = [None for _ in range(len_sum - len_validation - 1)] + [train_data[-1]] + validation_data.tolist()
        self.plot([padded_train_data, padded_validation_data], [f'{self.ticker} train', f'{self.ticker} validation'])

    def convert_time_series_data(self, data, time_step):
        batch_x = []
        batch_y = []
        for i in range(len(data) - time_step - 1):
            batch, _, _ = self.transform(data[i:(i + time_step + 1)])
            batch_x.append(batch[:time_step])
            batch_y.append(batch[-1])
        batch_x = np.asarray(batch_x).reshape((len(batch_x), time_step, 1)).astype('float32')
        batch_y = np.asarray(batch_y).reshape((len(batch_y), 1, 1)).astype('float32')
        return batch_x, batch_y

    def transform(self, arr):
        arr = np.asarray(arr, dtype=np.float32)
        max_val = np.max(arr[:self.time_step])
        arr /= np.max(arr)
        criteria = arr[0]
        arr -= criteria
        return arr, criteria, max_val

    def inverse_transform(self, arr, criteria, max_val):
        arr = np.asarray(arr, dtype=np.float32)
        arr += criteria
        arr *= max_val
        return arr

    def plot(self, datas, legend):
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
        self.model.summary()
        os.makedirs('checkpoints', exist_ok=True)
        optimizer = tf.keras.optimizers.Adam(lr=self.lr)
        iteration_count = 0
        print(f'\ntrain on {len(self.train_x)} samples. train_x.shape : {self.train_x.shape}')
        print(f'validate on {len(self.validation_x)} samples. validation_x.shape : {self.validation_x.shape} ')
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
                if iteration_count % 5000 == 0:
                    loss = self.evaluate_validation()
                    self.model.save(f'checkpoints/{self.ticker}_{iteration_count}_iter_{loss:.2f}_mae.h5', include_optimizer=False)
                if iteration_count == self.max_iteration_count:
                    break_flag = True
                    break
            if break_flag:
                break
        self.evaluate_train(show_plot=True)
        self.evaluate_validation(show_plot=True, future_step=self.future_step)

    def evaluate_train(self, show_plot=False, future_step=0):
        print()
        x = self.train_data[:self.time_step]
        y_seq = []
        for i in tqdm(range(len(self.train_data) - self.time_step + future_step)):
            x, criteria, max_val = self.transform(x)
            y = self.graph_forward(self.model, np.asarray(x).reshape((1, self.time_step, 1)), False)
            x = np.append(x[1:], np.asarray(y).reshape(-1)[-1])
            x = self.inverse_transform(x, criteria, max_val)
            y_seq.append(float(x[-1]))
            if self.time_step + i < len(self.train_data):
                x = np.append(x[:-1], self.train_data[self.time_step + i])
        y_true = np.asarray(self.train_data[self.time_step:])
        y_pred = np.asarray(y_seq)
        mae = -1.0
        if future_step == 0:
            mae = np.mean(np.abs(np.asarray(y_true - y_pred)))
            print(f'train data MAE : {mae:4f}\n')
        if show_plot:
            self.plot([y_true, y_pred], [f'{self.ticker}', 'AI predicted'])
        return mae

    def evaluate_validation(self, show_plot=False, future_step=0):
        print()
        x = list(self.train_data[-self.time_step:])
        y_seq = []
        forward_length = len(self.validation_data) + future_step
        for i in tqdm(range(forward_length)):
            x, criteria, max_val = self.transform(x)
            y = self.graph_forward(self.model, np.asarray(x).reshape((1, self.time_step, 1)), False)
            x = np.append(x[1:], np.asarray(y).reshape(-1)[-1])
            x = self.inverse_transform(x, criteria, max_val)
            y_seq.append(float(x[-1]))
            if self.time_step + i < len(self.validation_data):
                x = np.append(x[:-1], self.validation_data[self.time_step + i])

        y_true = np.asarray(self.validation_data)
        y_pred = np.asarray(y_seq)
        mae = np.mean(np.abs(np.asarray(y_true - y_pred[:len(y_true)])))
        print(f'validation MAE : {mae:4f}\n')
        if show_plot:
            if future_step > 0:
                y_true = y_true.tolist()
                y_pred = y_pred.tolist()
                y_pred_validation_only = y_pred[:len(y_true)]
                y_pred_predicted_future_only = y_pred[len(y_true):]
                padded_y_pred_predicted_future = [None for _ in range(len(y_true) - 1)] + [y_pred_validation_only[-1]] + y_pred[len(y_true):]
                print(f'end of data : {y_pred_validation_only[-1]:.4f}')
                for i in range(len(y_pred_predicted_future_only)):
                    print(f'predicted {i + 1} day after: {y_pred_predicted_future_only[i]:.4f}')
                self.plot([y_true, y_pred_validation_only, padded_y_pred_predicted_future], [f'{self.ticker}', 'AI predicted', 'AI predicted future'])
            else:
                self.plot([y_true, y_pred], [f'{self.ticker}', 'AI predicted'])
        return mae
