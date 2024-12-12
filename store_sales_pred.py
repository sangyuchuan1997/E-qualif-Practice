import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_log_error
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s : %(message)s',
    filename='app.log',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)


def load_data(dataset_path: Path) -> pd.DataFrame:
    logger.info("- start loading train.csv")
    train = pd.read_csv(str(Path.joinpath(dataset_path, 'train.csv')),
                        usecols=['store_nbr', 'family',
                                 'date', 'sales', 'onpromotion'],
                        dtype={'store_nbr': 'category',
                               'family': 'category',
                               'sales': 'float32',
                               'onpromotion': 'uint32',
                               },
                        parse_dates=['date'],
                        )
    logger.info("- end loading train.csv")
    train['date'] = train.date.dt.to_period('D')
    train = (train
             .set_index(['date', 'family', 'store_nbr'])
             .sort_index()
             )
    logger.info("- train data get grouped by DATE, FAMILY, STORE_NBR")

    logger.info("- start loading test.csv")
    test = pd.read_csv(str(Path.joinpath(dataset_path, "test.csv")),
                       dtype={'store_nbr': 'category',
                              'family': 'category',
                              'onpromotion': 'uint32',
                              },
                       parse_dates=['date'],
                       )
    logger.info("- end loading test.csv")
    test['date'] = test.date.dt.to_period('D')
    test = test.set_index(['date', 'family', 'store_nbr']).sort_index()
    logger.info("- test data get grouped by DATE, FAMILY, STORE_NBR")

    logger.info("- start loading holidays_events.csv")
    holidays_events = pd.read_csv(str(Path.joinpath(dataset_path, "holidays_events.csv")),
                                  dtype={'type': 'category',
                                         'locale': 'category',
                                         'locale_name': 'category',
                                         'description': 'category',
                                         'transferred': 'bool',
                                         },
                                  parse_dates=['date'],
                                  infer_datetime_format=True,
                                  )
    logger.info("- end loading holidays_events.csv")
    holidays_events = holidays_events.set_index('date').to_period('D')

    logger.info(f"Data: \n{test}")

    return


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    return df


def rmsle(y_true, y_pred):
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("RMSLE は負の値には対応していません")
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    squared_errors = (log_true - log_pred) ** 2

    rmsle = np.sqrt(np.mean(squared_errors))

    return rmsle


def main():
    logger.info("Start loading datasets.")
    dataset_path = Path("datasets_kaggle/store-sales-time-series-forecasting/")
    load_data(dataset_path)
    logger.info("End loading datasets.")

    # output_path = Path("output.csv")

    # データの処理
    # processed_df = process_data(df)

    # 結果の保存
    # processed_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    logger.info("===================")
    logger.info("Execution starts.")
    starttime = datetime.now()
    logger.info("Startime: %s", starttime.strftime('%Y-%m-%d %H:%M:%S'))
    try:
        main()
    except Exception as e:
        logger.error(f"Error occured: {e}", exc_info=True)
    finally:
        logger.info("Execution terminates.")
        endtime = datetime.now()
        logger.info("Endtime: %s", endtime.strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("Execution time: %s", endtime - starttime)
        logger.info("===================")
