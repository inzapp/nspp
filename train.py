from nspp import NasdaqStockPricePredictor as NSPP

if __name__ == '__main__':
    NSPP(
        ticker='AMZN',
        start_date='2021-01-01',
        end_date='2022-01-01',
        lr=0.01,
        time_step=14,
        batch_size=32,
        future_step=7,
        validation_ratio=0.2,
        max_iteration_count=100).fit()
