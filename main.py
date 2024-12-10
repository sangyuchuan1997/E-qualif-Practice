import logging

# ログの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',
    encoding='utf-8'
)
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info('プログラムを開始します')
        # メインの処理をここに書く
        logger.info('プログラムを正常終了します')
    except Exception as e:
        logger.error(f'エラーが発生しました: {str(e)}')
        raise

if __name__ == "__main__":
    main()