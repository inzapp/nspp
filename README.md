# NSPP(Nasdaq Stock Price Predictor)
NSPP is Nasdaq stock price predictor using LSTM model.

If you have your custom time series data, you can use it even if it is not stock price data.

## Usage
Edit train.py
```python
NSPP(
    ticker='AMZN',
    start_date='2020-01-01',
    end_date='2022-01-01',
    interval='1d',  # available interval : '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
    lr=0.001,
    time_step=14,
    batch_size=32,
    future_step=7,
    validation_ratio=0.2,
    max_iteration_count=5000).fit()
```

If you want to use your custom data, just input your data file name into ticker.
```python
ticker='sample_data.txt'  # test with built in sample data sample_data.txt
```

Run train.py
```
python train.py
```

## Result
First, train data with validation data are shown with different color before start training.

You can change validation data ratio with validation_ratio in train.py

<img src="/md/sample_train_validation_data.jpg" width="800"><br>

Model is saved with validation MAE and MAPE every 2000 iterations.

```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 379/379 [00:00<00:00, 954.75it/s]
0it [00:00, ?it/s]
validation MAPE : 0.0090, MAE : 3.779435

[  4000 iter] loss => 0.0069
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 379/379 [00:00<00:00, 1067.52it/s]
0it [00:00, ?it/s]
validation MAPE : 0.0088, MAE : 3.685025

[  6000 iter] loss => 0.0074
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 379/379 [00:00<00:00, 1088.94it/s]
0it [00:00, ?it/s]
validation MAPE : 0.0089, MAE : 3.749943

[  8000 iter] loss => 0.0068
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 379/379 [00:00<00:00, 1038.38it/s]
0it [00:00, ?it/s]
validation MAPE : 0.0088, MAE : 3.688454

[ 10000 iter] loss => 0.0067
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 379/379 [00:00<00:00, 1001.28it/s]
0it [00:00, ?it/s]
validation MAPE : 0.0087, MAE : 3.647736


100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1500/1500 [00:01<00:00, 1034.62it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00, 999.53it/s]

$ ls -alrt ./checkpoints/
-rw-r--r-- 1 inzapp 197121 91640  7월 10 18:38 sample_data.txt_0.2_val_2000_iter_0.0290_MAPE_220.0937_MAE_cd.h5
-rw-r--r-- 1 inzapp 197121 91640  7월 10 18:38 sample_data.txt_0.2_val_4000_iter_0.0235_MAPE_178.4185_MAE_cd.h5
-rw-r--r-- 1 inzapp 197121 91640  7월 10 18:39 sample_data.txt_0.2_val_2000_iter_0.0292_MAPE_221.4532_MAE_cd.h5
-rw-r--r-- 1 inzapp 197121 91640  7월 10 18:39 sample_data.txt_0.2_val_4000_iter_0.0236_MAPE_179.1997_MAE_cd.h5
drwxr-xr-x 1 inzapp 197121     0  7월 10 18:46 ../
drwxr-xr-x 1 inzapp 197121     0  7월 10 18:55 ./
```

After training end, model predicted result is shown with future prediction.

Train data prediction

<img src="/md/sample_train_data_with_prediction.jpg" width="800"><br>

Validation data prediction

<img src="/md/sample_validation_data_with_prediction.jpg" width="800"><br>

## Real ticker from Nasdaq market

You can use real ticker name from Nasdaq market with yfinance API.

```python
ticker='SPY'
```
Train data prediction

<img src="/md/spy_train_data_with_prediction.jpg" width="800"><br>

Validation data prediction

<img src="/md/spy_validation_data_with_prediction.jpg" width="800"><br>
