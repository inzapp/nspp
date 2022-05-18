from nspp import NasdaqStockPricePredictor as NSPP

if __name__ == '__main__':
    NSPP(
        ticker='sample_data',
        start_date='2020-01-01',
        end_date='2022-01-01',
        interval='1d',  # available interval : '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        lr=0.01,
        time_step=14,
        batch_size=32,
        future_step=7,
        validation_ratio=0.2,
        max_iteration_count=5000).fit()
