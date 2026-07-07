# task_1_func_1.py
# 主要用于：从 Wikipedia 获取 S&P 500 成分股列表，及使用 Tiingo 包下载价格数据
import time
import random
import pandas as pd
import requests
from io import StringIO
from tiingo import TiingoClient

def data_download():
    # 从 Wikipedia 获取 S&P 500 成分股列表
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        sp500 = tables[0]
        tickers = sp500['Symbol'].tolist()
        print(f"成功从 Wikipedia 获取 {len(tickers)} 只股票")
    except Exception as e:
        print(f"从 Wikipedia 获取失败: {e}")
        print("使用给定备选列表")
        # 备选列表（前100只左右的常见股票）
        tickers = [
            "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
            "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
            "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
            "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
            "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
            "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",
            "BDX", "BRK.B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA",
            "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF.B", "BLDR", "BG", "BXP",
            "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CVNA", "CAT",
            "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR"
        ]

    # 只取前100个股票代码
    tickers = tickers[:100]

    # 将代码中的点号替换为连字符，以符合 Tiingo 格式
    tickers = [t.replace('.', '-') for t in tickers]
    print(f"最终使用 {len(tickers)} 只股票:\n{tickers}")

    # 配置 Tiingo API 密钥
    config = {'api_key': "KEY", 'session': True}
    client = TiingoClient(config)

    start_date = '2016-01-01'
    end_date = '2025-12-31'

    # 定义下载单只股票的函数，支持重试
    def download_single_ticker(ticker_1, max_retries=3):
        for attempt in range(max_retries):
            try:
                series_1 = client.get_dataframe(ticker_1,
                                                startDate=start_date,
                                                endDate=end_date,
                                                frequency='daily',
                                                metric_name='adjClose')
                if series_1 is not None and not series_1.empty:
                    series_1.name = ticker_1
                    return series_1
                else:
                    print(f"  {ticker_1} 返回空数据")
                    return None
            except Exception as e_1:
                print(f"    第 {attempt + 1} 次下载 {ticker_1} 失败: {e_1}")
                if "rate limit" in str(e_1).lower() or "too many requests" in str(e_1).lower():
                    wait_time = 60 * (2 ** attempt)  # 限流等待递增
                    print(f"    触发限流,等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    wait_time = 10
                    print(f"    等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        print(f"  {ticker_1} 最终失败, 跳过")
        return None

    successful_series = []
    total = len(tickers)

    # 依次下载每只股票，每下载完一只随机等待60~120秒以避免API限制
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{total}] 下载 {ticker} ...")
        series = download_single_ticker(ticker)
        if series is not None and not series.empty:
            successful_series.append(series)
            print(f"    成功获取 {len(series)} 个交易日数据")
        else:
            print(f"    无数据, 跳过")

        sleep_time = random.uniform(60, 120)
        print(f"    等待 {sleep_time:.1f} 秒后继续...")
        time.sleep(sleep_time)

    # 合并所有成功下载的股票数据，保存为CSV
    if successful_series:
        adj_close = pd.concat(successful_series, axis=1)
        adj_close.to_csv('original_data.csv')
        print(f"\n原始数据保存成功!")
        print(f"最终股票数量: {adj_close.shape[1]}")
        print(f"时间范围: {adj_close.index[0]} 到 {adj_close.index[-1]}")
        print(f"保存文件: original_data.csv")
    else:
        print("\n错误: 没有下载到任何数据")

    return 0