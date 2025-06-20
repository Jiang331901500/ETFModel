import torch
import numpy as np
import pandas as pd
import akshare as ak
from sklearn.preprocessing import RobustScaler
from transformers import pipeline
import time
import threading
import news_fetch_functions
from mongo_db_manager import MongoDBManager
from stop_input import StopInput

class DataPreprocessor:
    etf_df: pd.DataFrame
    news_df: pd.DataFrame
    etf_holdings_dict: dict # 用于存储ETF持仓成分股 {code: {year: set(holdings)}}

    industry_keywords = {
        # 金融行业
        '银行': ['央行', '存贷比', '不良率', '拨备率', 'LPR', '资本充足率', '同业拆借', '银保监会', '存款准备金'],
        '非银金融': ['券商', '保费', '保单', '投行', '两融', '资管新规', '险资', '科创板', '注册制', 'IPO'],
        # 周期行业
        '房地产': ['土拍', '去化率', '限购', '房贷利率', '棚改', '三道红线', '预售资金', '保障房'],
        '建筑装饰': ['基建', 'PPP', 'EPC', '装配式', '钢结构', '工程款', '招投标', '一带一路'],
        '钢铁': ['粗钢', '吨钢利润', '高炉开工率', '螺纹钢', '铁矿石', '去产能', '库存周期'],
        '有色金属': ['电解铝', '锂矿', '钴价', '稀土', '贵金属', '伦铜', '库存', '加工费'],
        '化工': ['PTA', 'PX', 'MDI', '钛白粉', '有机硅', '炼化', '产能利用率', '价差'],
        # 消费行业
        '食品饮料': ['白酒', '提价', '动销', '库存', '奶价', '调味品', '餐饮渠道', '高端化'],
        '家用电器': ['白电', '黑电', '小家电', '能效标准', '以旧换新', '零售额', '线上占比'],
        '农林牧渔': ['猪周期', '存栏量', '能繁母猪', '鸡苗', '饲料', '转基因', '中央一号文件'],
        '汽车': ['乘用车', '新能源车', '排产', '电池', '自动驾驶', '零部件', '缺芯'],
        '商贸零售': ['社零', '线上渗透率', '直播带货', '免税', '跨境电商', '会员店', 'GMV'],
        # 科技成长
        '电子': ['半导体', '晶圆厂', '封测', '消费电子', 'PCB', 'MLCC', '面板', '国产替代'],
        '计算机': ['信创', '云计算', '数据中心', '工业软件', '网络安全', '数字货币', '人工智能'],
        '通信': ['5G', '光模块', '基站', '物联网', '专网', '卫星互联网', '运营商'],
        '传媒': ['游戏版号', '影视', '广告', '短视频', 'AR/VR', '元宇宙', '版权'],
        '医药生物': ['集采', '创新药', '临床试验', '生物药', 'CXO', '医疗器械', 'DRG'],
        '电力设备': ['光伏', '硅料', '组件', '逆变器', '风电', '招标', '储能', '特高压'],
        # 公共事业
        '公用事业': ['电价', '煤电联动', '燃气', '水价', '垃圾发电', '碳交易', '绿电'],
        '交通运输': ['货运', '集装箱', 'BDI', '快递', '航空', '高铁', '油价', '免税店'],
        '环保': ['碳中和', '碳达峰', '污水处理', '固废', '再生资源', 'ESG'],
    }
    MACRO_KEYWORDS = {
        # ======== 货币政策类 (权重系数: 1.0-1.8) ========
        "降准": 1.8, "降息": 1.8, "存款准备金": 1.5, "MLF": 1.4, "中期借贷便利": 1.4,
        "逆回购": 1.3, "公开市场操作": 1.3, "利率走廊": 1.2, "LPR": 1.6, "贷款市场报价利率": 1.6,
        "货币政策": 1.4, "货币供应": 1.3, "M2": 1.3, "社会融资": 1.5, "信贷政策": 1.3,
        "流动性": 1.4, "资金面": 1.2, "基础货币": 1.1, "再贷款": 1.2, "再贴现": 1.1,
        
        # ======== 财政政策类 (权重系数: 1.0-1.7) ========
        "财政政策": 1.5, "财政赤字": 1.6, "财政刺激": 1.7, "减税": 1.7, "减费": 1.5,
        "专项债": 1.6, "地方债": 1.5, "国债": 1.3, "财政支出": 1.4, "基建投资": 1.6,
        "转移支付": 1.2, "预算内": 1.2, "财政存款": 1.1, "PPP": 1.3, "政府购买": 1.3,
        
        # ======== 监管政策类 (权重系数: 1.2-2.0) ========
        "证监会": 1.8, "银保监会": 1.7, "央行": 1.9, "人民银行": 1.9, "金融监管": 1.7,
        "注册制": 2.0, "退市制度": 1.8, "IPO": 1.7, "再融资": 1.6, "减持新规": 1.7,
        "资管新规": 1.9, "北交所": 1.6, "科创板": 1.7, "创业板": 1.5, "交易规则": 1.4,
        "窗口指导": 1.8, "政策风险": 1.6, "监管政策": 1.5, "金融稳定": 1.6, 
        "金融开放": 1.5, "金融创新": 1.4, "金融科技": 1.3,
        
        # ======== 国际经济类 (权重系数: 1.2-2.2) ========
        "美联储": 2.0, "FOMC": 2.0, "加息": 2.2, "缩表": 2.0, "量化宽松": 1.9,
        "欧央行": 1.7, "日本央行": 1.5, "英国央行": 1.5, "关税": 2.0, "贸易战": 2.2,
        "贸易摩擦": 2.1, "贸易协定": 1.9, "WTO": 1.6, "汇率": 1.9, "人民币": 1.8,
        "美元指数": 1.7, "外汇储备": 1.6, "资本流动": 1.7, "离岸人民币": 1.7,
        "SWIFT": 1.5, "国际清算": 1.4,
        
        # ======== 宏观经济指标类 (权重系数: 1.3-1.8) ========
        "GDP": 1.8, "国内生产总值": 1.8, "CPI": 1.7, "PPI": 1.7, "通胀": 1.8,
        "通缩": 1.8, "PMI": 1.7, "采购经理指数": 1.7, "工业增加值": 1.6, "固定资产投资": 1.7,
        "社会消费品零售": 1.6, "失业率": 1.8, "就业": 1.7, "用电量": 1.4, "货运量": 1.4,
        "景气指数": 1.5, "先行指标": 1.5, "克强指数": 1.6, "经济增长": 1.7,
        "经济周期": 1.6, "宏观经济": 1.5, "宏观调控": 1.5, 
        
        # ======== 行业政策类 (权重系数: 1.1-1.7) ========
        "房地产政策": 1.7, "房住不炒": 1.7, "限购": 1.6, "限贷": 1.6, "三道红线": 1.7,
        "碳中和": 1.7, "碳达峰": 1.6, "新能源": 1.6, "双碳目标": 1.6, "供给侧改革": 1.7,
        "去杠杆": 1.7, "产业政策": 1.5, "反垄断": 1.8, "平台经济": 1.6, "专精特新": 1.5,
        
        # ======== 市场情绪类 (权重系数: 1.2-1.8) ========
        "市场风险": 1.6, "系统性风险": 1.8, "黑天鹅": 1.8, "灰犀牛": 1.7, "流动性风险": 1.7,
        "信用风险": 1.6, "杠杆率": 1.5, "爆仓": 1.8, "平仓线": 1.6, "融资融券": 1.5,
        "北向资金": 1.7, "南向资金": 1.5, "主力资金": 1.4, "机构持仓": 1.4,

        # ======== ETF\指数类 (权重系数: 1.2-1.8) ========
        "ETF": 1.5, "指数基金": 1.4, "沪深300": 1.6, "上证50": 1.5, "创业板": 1.5, "科创板": 1.5, 
        "中证500": 1.5, "ETF联接": 1.4, "指数增强": 1.3, 
        "深成指": 1.5, "沪指": 1.5, 
        "恒生指数": 1.5, "纳斯达克": 1.6, "标普500": 1.5,
        
        # ======== 特殊事件类 (权重系数: 1.8-2.5) ========
        "金融危机": 2.5, "经济危机": 2.4, "债务危机": 2.3, "疫情": 2.2, "公共卫生": 2.0,
        "地缘政治": 2.3, "战争": 2.5, "制裁": 2.4, "天灾": 2.0, "重大事故": 1.9,
        "政策转向": 2.2, "领导讲话": 2.0, "中央会议": 2.3, "国务院会议": 2.2
    }
    authority_sources = {
        '财新网': 1.0,
    }

    def __init__(self, config: dict, tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.etf_code = config['etf_code']
        self.etf_holdings_dict = {}
        try:
            self.mongo_mgr = MongoDBManager()
        except Exception as e:
            print("[WARN] Mongo DB not enable!")

        # 加载预训练金融情感模型
        self.sentiment_analyzer = pipeline(
            "text-classification", 
            model="yiyanghkust/finbert-tone-chinese",
            tokenizer="yiyanghkust/finbert-tone-chinese"
        )

        self.stopper = StopInput()

    def load_data_frame(self, from_csv = False) -> bool:
        etf_csv_path = f"{self.config['data_dir']}/{self.etf_code}_etf.csv"
        news_csv_path = f"{self.config['data_dir']}/{self.etf_code}_news.csv"
        if from_csv: # 直接从csv中加载数据
            try:
                self.etf_df = pd.read_csv(etf_csv_path, parse_dates=['date', 'insert_time'], index_col='date', dtype={'code': str})
                self.news_df = pd.read_csv(news_csv_path, parse_dates=['insert_time'], dtype={'title': str, 'content': str})
                self.news_df['title'] = self.news_df['title'].fillna("")
                self.news_df['content'] = self.news_df['content'].fillna("")
                print(f"Loaded ETF data from {etf_csv_path} and news data from {news_csv_path}.")
                return True
            except Exception as e:
                print(f"Failed to load from CSV: {e}, please try to fetch from network. (from_csv=False)")
            return False
    
        # 非csv加载数据，从网络获取数据并保存到MongoDB
        self.fetch_etf_data()
        self.fetch_news()
        # etf 和 news 的日期取交集
        news_dates = pd.to_datetime(self.news_df['date'], format='%Y-%m-%d').sort_values().values
        self.etf_df = self.etf_df[(self.etf_df['date'] >= news_dates[0]) & (self.etf_df['date'] <= news_dates[-1])]

        # 总是保存最新数据到csv文件中以方便部署到服务器使用
        self.etf_df.to_csv(etf_csv_path, index=True)
        self.news_df.to_csv(news_csv_path, index=False)
        return True

    def fetch_etf_data(self):
        self.incremental_fetch_etf()
        # 从MongoDB加载ETF数据
        print("Loading ETF data from MongoDB...")
        self.etf_df = self.mongo_mgr.load_etf_data(
            code=self.etf_code
        )
        if self.etf_df.empty:
            print("No ETF data found in the specified date range.")
            return
        # 计算技术指标
        self.prepare_etf_data()
        # 获取基金持仓
        self.get_etf_holdings()

    def fetch_news(self):
        threads = []
        start_date = self.etf_df.index.min()
        end_date = self.etf_df.index.max() + pd.DateOffset(days=1)
        while True:
            incremental_count = 0
            # 获取新闻数据
            for source_name, source_func in news_fetch_functions.function_dict.items():
                print(f"--- Fetching news from {source_name}...")
                incremental_count += self.incremental_fetch_news(
                    source_func=source_func, 
                    source_name=source_name
                )

            # 等待之前的线程完成
            if threads:
                print(f"***[DEBUG] Waiting for previous {len(threads)} threads to finish...")
                for thread in threads:
                    thread.join()
                threads.clear()  # 清空线程列表，准备下一轮增量处理

            if incremental_count == 0:
                print("!!! Exiting news fetching outer-loop.")
                break
            
            if self.stopper.should_stop():
                self.stopper.clean_resources()
                print("!!! Stopping news fetching as stop flag is set.")
                break

             # Create and run the processing thread
            def process_news():
                for source_name in news_fetch_functions.function_dict:
                    # 从MongoDB加载所有原始新闻数据
                    print(f"### Loading all {source_name} news data from MongoDB within date range:", start_date, "to", end_date)
                    news_df = self.mongo_mgr.load_news(
                        start_date=start_date, 
                        end_date=end_date,
                        sources=[source_name]
                    )
                    if news_df.empty:
                        print(f"### No {source_name} news data found in the specified date range.")
                        continue
                    print(f"### Totally loaded {len(news_df)} {source_name} news records from MongoDB to process.")

                    # 增量计算新闻得分
                    # 对于last_date, earliest_date之外的新闻数据
                    last_date, earliest_date = self.mongo_mgr.get_latest_and_earliest_featured_news_date(source_name)
                    news_df = news_df[(news_df['date'] > last_date) | (news_df['date'] < earliest_date)]
                    print(f"### Filtered news data to {len(news_df)} {source_name} records for processing.")
                                    
                    # 分块进行处理，以便于统计进度
                    processed_count = 0
                    while (not self.stopper.should_stop()) and (processed_count < len(news_df)):
                        batch_size = 200  # 每次处理200条新闻
                        # 剩余不足一批时，直接处理剩余数据
                        if (processed_count + batch_size) > len(news_df):
                            batch_size = len(news_df) - processed_count
                        batch_df = news_df.iloc[processed_count : processed_count + batch_size]
                        print(f"### Processing {source_name} batch {processed_count // batch_size + 1} of {len(news_df) // batch_size + 1}, batch size: {batch_size}")
                        inc_news = self.generate_news_score(batch_df)
                        if len(inc_news) == batch_size:
                            # 将增量新闻得分数据插入MongoDB
                            inserted_count = self.mongo_mgr.insert_featured_news(inc_news)
                            print(f"@@@ Inserted {inserted_count} {source_name} new featured news records into MongoDB.")
                            processed_count += batch_size

            process_thread = threading.Thread(target=process_news)
            process_thread.start()
            threads.append(process_thread)
        
        # 从MongoDB加载所有处理过的新闻
        self.news_df = self.mongo_mgr.load_featured_news(
            start_date=start_date, 
            end_date=end_date
        )
        if self.news_df.empty:
            print("No featured news data found in the specified date range.")
            return
        print(f"Loaded {len(self.news_df)} featured news records from MongoDB.")
        self.news_df['date'] = self.news_df['date'].dt.strftime('%Y-%m-%d')

    def incremental_fetch_etf(self):
        """增量获取ETF数据主函数"""
        code = self.etf_code
        # 获取最新和最早日期
        last_date, earliest_date = self.mongo_mgr.get_latest_and_earliest_etf_date(code)
        start_date = last_date + pd.DateOffset(days=1) # 从最后日期的下一天开始获取数据
        end_date = pd.Timestamp.now() - pd.DateOffset(days=1)  # 获取到昨天的数据
        if start_date > end_date:
            print(f"没有新的ETF数据需要获取，最后日期: {last_date}, 当前日期: {end_date}")
            return
        print(f"开始获取 {start_date} 之后的 {code} ETF 数据")
        
        # 获取最新数据
        new_df = ak.fund_etf_hist_em(
            symbol=code,
            period="daily",
            start_date=start_date.strftime('%Y%m%d'),
            end_date=end_date.strftime('%Y%m%d'),
            adjust="qfq"
        )
        
        if new_df.empty:
            print("未获取到新的ETF数据。")
            return
        
        new_df = new_df.rename(columns={'日期': 'date', '收盘': 'close', '成交量': 'vol', '成交额': 'amount', '涨跌幅': 'delta'})
        new_df['code'] = code
        new_df['date'] = pd.to_datetime(new_df['date'], format='%Y-%m-%d')
        new_df = new_df[['date', 'code', 'close', 'vol', 'amount', 'delta']]
        
        if not new_df.empty:
            inserted_count = self.mongo_mgr.insert_etf_data(new_df)
            print(f"插入 {self.etf_code} ETF 数据 {inserted_count} 条")
        else:
            print("没有新的ETF数据需要插入。")

    def incremental_fetch_news(self, source_func, source_name):
        """增量获取新闻主函数"""
        last_date, earliest_date = self.mongo_mgr.get_latest_and_earliest_news_date(source_name)
        print(f"--- 开始获取 {source_name} 新闻，当前库中最后日期: {last_date}, 当前库中最早日期: {earliest_date}")
        
        inserted_count = 0
        total_size = 0
        while True:
            # 获取该来源最新日期
            last_date, earliest_date = self.mongo_mgr.get_latest_and_earliest_news_date(source_name)
            # 获取增量新闻数据
            news_df = source_func(total_size=total_size, batch_size=200, last_date=last_date, earliest_date=earliest_date)
            # 无数据时停止
            if news_df.empty and total_size > 0:
                break
        
            inserted_count += self.mongo_mgr.insert_news(news_df, source_name)
            total_size = self.mongo_mgr.get_news_count(source_name)
            print(f"@@@ 已插入 {inserted_count} 条 {source_name} 新闻数据，当前总数: {total_size}")

            # 此处限制一下每次获取的新闻数量，避免一次性获取过多数据
            if inserted_count >= 200:
                break

        if inserted_count > 0:
            print(f"--- 成功获取 {source_name} 的新数据，共计 {inserted_count} 条。")
        else:
            print(f"--- 未获取到更多 {source_name} 的新数据。")

        return inserted_count

    # 计算技术指标
    def calculate_technical_indicators(self):
        df = self.etf_df
        # 移动平均
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # MACD
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 波动率
        df['volatility'] = df['close'].rolling(20).std()
        
        # 量能变化率
        df['vol_change'] = df['vol'].pct_change()

        self.etf_df = df.dropna()  # 删除包含NaN的行

    # 数据标准化
    def scale_features(self):
        window = self.config['sample_sequence_window']
        df = self.etf_df.copy()
        """滑动窗口标准化，避免未来信息泄露"""
        features = ['close', 'vol', 'amount', 'ma5', 'ma20', 'ma60', 
                'macd', 'signal', 'rsi', 'volatility', 'vol_change']
        
        scaled_data = np.zeros_like(df[features])
        scaler = RobustScaler()
        
        for i in range(window, len(df)):
            scaler.fit(df.iloc[i-window:i][features])
            scaled_data[i] = scaler.transform(df.iloc[i:i+1][features])
        
        return pd.DataFrame(scaled_data, index=df.index, columns=features)

    # 主数据处理函数
    def prepare_etf_data(self):
        # etf数据以date为索引，按照date排升序
        self.etf_df.index = self.etf_df['date']
        self.etf_df = self.etf_df.sort_index()
        self.calculate_technical_indicators()
        # 使用列delta/100来计算未来5日的涨跌幅作为目标值target_{i}
        for i in range(1, 6):
            self.etf_df[f'target_{i}'] = (self.etf_df['delta'].shift(-i) / 100.0).round(4)
    
    # 获取ETF持仓成分股词典
    def get_etf_holdings(self):
        code = self.etf_code

        # 如果持仓数据已存在，则直接加载
        holdings_file = f"{self.config['data_dir']}/etf_holdings.pkl"
        try:
            self.etf_holdings_dict = pd.read_pickle(holdings_file)
            print(f"Loaded ETF holdings data from {holdings_file}.")
        except FileNotFoundError:
            # 使用akshare获取各年份的持仓
            print("Fetching ETF holdings data...")

        if code not in self.etf_holdings_dict:
            self.etf_holdings_dict[code] = {}
        
        this_year = pd.Timestamp.now().year
        for year in range(2013, this_year + 1):
            fetch_year = str(year)
            if fetch_year in self.etf_holdings_dict[code]:
                continue
            try:
                holdings_df = ak.fund_portfolio_hold_em(symbol=code, date=fetch_year)
            except Exception as e:
                holdings_df = pd.DataFrame()
            time.sleep(0.2)  # 避免请求过快
            if holdings_df.empty:
                print(f"No holdings data found for {code} in {year}.")
                continue
            # 将持仓数据存入字典
            self.etf_holdings_dict[code][fetch_year] = set(holdings_df.head(50)['股票名称'].tolist())

        # 将self.etf_holdings_dict保存到本地数据文件
        holdings_file = f"{self.config['data_dir']}/etf_holdings.pkl"
        pd.to_pickle(self.etf_holdings_dict, holdings_file)

    def extract_news_features(self, news_row):
        text = news_row['title'] + ' ' + news_row['content']
        year = str(news_row['date'].year)
        holdings = self.etf_holdings_dict.get(self.etf_code, set()).get(year, set())
        
        # 1. 成分股关联度
        stock_relation = sum(1 for stock in holdings if stock in text)
        stock_relation = min(stock_relation, 3)  # 上限 3 分
        
        # 2. 行业关联度（使用TF-IDF）
        industry_score = 1
        # for industry, words in self.industry_keywords.items():
        #     industry_score += sum(text.count(word) for word in words)
        
        # 3. 情感强度（使用预训练模型
        try:
            key_text = text[:512] if len(text) > 512 else text
            result = self.sentiment_analyzer(key_text)[0]
            # 情感映射：Positive->1, Negative->1, Neutral->0
            sentiment_map = {"Positive": 1, "Negative": 1, "Neutral": 0}
            sentiment_strength = sentiment_map[result['label']] * result['score']
        except:
            sentiment_strength = 0  # 分析失败时返回中性
        
        # 4. 信源权重（预设权威媒体列表）
        source_weight = self.authority_sources.get(news_row['source'], 0.5) if 'source' in news_row else 0.5
        
        # 5. 基于关键词的关联度分析
        keyword_score = 0
        for kw, weight in self.MACRO_KEYWORDS.items():
            if kw in text:
                keyword_score += weight * (1 + text.count(kw)*0.2)
        keyword_score = min(keyword_score, 3)  # 上限 3 分
        
        return {
            'stock_relation': stock_relation,
            'industry_score': industry_score,
            'sentiment': sentiment_strength,
            'source_weight': source_weight,
            'keyword_score': keyword_score
        }
    
    def calculate_news_score(self, features):
        weights = {
            'stock_relation': 0.3,   # 成分股关联最重要
            'industry_score': 0.1,
            'sentiment': 0.2,
            'source_weight': 0.1,
            'keyword_score': 0.3
        }
        return sum(features[k] * weights[k] for k in features)

    def generate_news_score(self, news_df: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            return news_df
        # 特征提取和分数计算
        print("Calculating news scores...")
        # 对每一行计算features和score
        news_df['features'] = news_df.apply(self.extract_news_features, axis=1)
        news_df['score'] = news_df['features'].apply(self.calculate_news_score)
        return news_df

    def preprocess_news(self):
        """
        预处理新闻数据，包含权重计算
        返回:
            processed_news: 字典 {date: {
                'input_ids': [max_news_per_day, max_length],
                'attention_mask': [max_news_per_day, max_length],
                'news_weights': [max_news_per_day]
            }}
        """

        # news_df应包含列: date, title, content, score
        max_news_per_day = self.config['max_news_per_day']
        max_length = self.config['max_news_length']
        # 按日期分组
        grouped = self.news_df.groupby('date')
        processed_news = {}
        
        for date, group in grouped:
            # 按重要性排序并选择最重要的新闻
            group = group.sort_values('score', ascending=False)
            daily_news = group.head(max_news_per_day)
            
            # softmax 适当放大差距，温度系数3
            weights = torch.softmax(torch.tensor(daily_news['score'].values * 3), dim=0)
            daily_news['weight'] = weights.numpy()
                
            # 填充空新闻
            if len(daily_news) < max_news_per_day:
                empty_rows = max_news_per_day - len(daily_news)
                empty_df = pd.DataFrame({
                    'title': [''] * empty_rows,
                    'content': [''] * empty_rows,
                    'score': [0] * empty_rows,
                    'weight': [0] * empty_rows
                })
                daily_news = pd.concat([daily_news, empty_df], ignore_index=True)
            
            # 文本预处理
            input_ids_list = []
            attention_mask_list = []
            weights_list = []
            
            for _, row in daily_news.iterrows():
                # 拼接标题和内容, 如果是空新闻则填充空字符串
                text = f"{row['title']} [SEP] {row['content']}"[:500] if row['title'] or row['content'] else ""
                tokens = self.tokenizer(
                    text, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids_list.append(tokens['input_ids'])
                attention_mask_list.append(tokens['attention_mask'])
                weights_list.append(row['weight'])
            
            # 堆叠所有新闻
            input_ids = torch.cat(input_ids_list, dim=0)  # [max_news_per_day, max_length]
            attention_mask = torch.cat(attention_mask_list, dim=0)  # [max_news_per_day, max_length]
            news_weights = torch.tensor(weights_list, dtype=torch.float32)  # [max_news_per_day]

            processed_news[date] = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'news_weights': news_weights
            }

        # 对于etf_df中存在的日期，确保每个日期都有对应的新闻数据，如果没有，则填充空新闻
        all_dates = set(self.etf_df.index.tolist())
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in processed_news:
                # 创建空新闻
                empty_tokens = self.tokenizer(
                    "", 
                    padding='max_length', 
                    truncation=True, 
                    max_length=max_length,
                    return_tensors='pt'
                )
                processed_news[date_str] = {
                    'input_ids': empty_tokens['input_ids'].repeat(max_news_per_day, 1),
                    'attention_mask': empty_tokens['attention_mask'].repeat(max_news_per_day, 1),
                    'news_weights': torch.zeros(max_news_per_day, dtype=torch.float32)
                }
        
        return processed_news

