from pymongo import MongoClient, ASCENDING
from datetime import datetime
import pandas as pd

class MongoDBManager:
    def __init__(self, db_name='quant_data', news_collection='news', featured_news_collection='featured_news', etf_collection='etf'):
        # 连接字符串（无密码本地连接）
        self.client = MongoClient('mongodb://localhost:27017/')
        self.db = self.client[db_name]
        self.news_collection = self.db[news_collection]
        self.etf_collection = self.db[etf_collection]
        self.featured_news_collection = self.db[featured_news_collection]
        
        # 创建索引（加速查询）
        self.news_collection.create_index([("date", ASCENDING)])
        self.news_collection.create_index([("source", ASCENDING), ("date", ASCENDING)])
        self.featured_news_collection.create_index([("date", ASCENDING)])
        self.featured_news_collection.create_index([("source", ASCENDING), ("date", ASCENDING)])
        self.etf_collection.create_index([("date", ASCENDING)])
        self.etf_collection.create_index([("code", ASCENDING), ("date", ASCENDING)])

        self.delete_invalid_etf_data() # 初始化时删除无效的ETF数据

    def clear_collections(self):
        """清空新闻和ETF数据集合"""
        self.news_collection.delete_many({})
        self.etf_collection.delete_many({})
        self.featured_news_collection.delete_many({})
        print("Collections cleared.")

    def get_news_count(self, source=None):
        """获取新闻数量"""
        if source:
            return self.news_collection.count_documents({"source": source})
        return self.news_collection.count_documents({})
    
    def get_featured_news_count(self, source=None):
        """获取处理后的新闻数量"""
        if source:
            return self.featured_news_collection.count_documents({"source": source})
        return self.featured_news_collection.count_documents({})
    
    def get_latest_and_earliest_news_date(self, source):
        """获取指定来源的最新和最早日期"""
        latest = self.news_collection.find_one(
            {"source": source}, 
            sort=[("date", -1)], 
            projection={"date": 1}
        )
        earliest = self.news_collection.find_one(
            {"source": source}, 
            sort=[("date", 1)], 
            projection={"date": 1}
        )
        latest_date = latest['date'] if latest else datetime(2000, 1, 1)
        earliest_date = earliest['date'] if earliest else datetime(2100, 1, 1)
        return latest_date, earliest_date

    def get_latest_and_earliest_featured_news_date(self, source):
        """获取精选新闻的最新和最早日期"""
        latest = self.featured_news_collection.find_one(
            {"source": source}, 
            sort=[("date", -1)], 
            projection={"date": 1}
        )
        earliest = self.featured_news_collection.find_one(
            {"source": source}, 
            sort=[("date", 1)], 
            projection={"date": 1}
        )
        latest_date = latest['date'] if latest else datetime(2000, 1, 1)
        earliest_date = earliest['date'] if earliest else datetime(2100, 1, 1)
        return latest_date, earliest_date
    
    def get_latest_and_earliest_etf_date(self, code):
        """获取指定ETF代码的最新和最早日期"""
        latest = self.etf_collection.find_one(
            {"code": code}, 
            sort=[("date", -1)], 
            projection={"date": 1}
        )
        earliest = self.etf_collection.find_one(
            {"code": code}, 
            sort=[("date", 1)], 
            projection={"date": 1}
        )
        latest_date = latest['date'] if latest else datetime(2000, 1, 1)
        earliest_date = earliest['date'] if earliest else datetime(2100, 1, 1)
        return latest_date, earliest_date
    
    def delete_invalid_etf_data(self):
        # 删除无效的ETF数据（日期$gte今天）
        today = pd.Timestamp.now().replace(hour=0, minute=0, second=0, microsecond=0)
        self.etf_collection.delete_many({"date": {"$gte": today}})
    
    def insert_news(self, news_df, source):
        """插入新数据并去重"""
        if news_df.empty:
            return 0
            
        # 添加数据来源标记
        news_df['source'] = source
        news_df['insert_time'] = datetime.now()
        
        # 批量插入（避免重复）
        records = news_df.to_dict('records')
        result = self.news_collection.insert_many(records, ordered=False)
        return len(result.inserted_ids)
    
    def insert_featured_news(self, featured_news_df):
        """插入精选新闻数据并去重"""
        if featured_news_df.empty:
            return 0
            
        # 添加插入时间
        featured_news_df['insert_time'] = datetime.now()
        
        # 批量插入（避免重复）
        records = featured_news_df.to_dict('records')
        result = self.featured_news_collection.insert_many(records, ordered=False)
        return len(result.inserted_ids)
    
    def insert_etf_data(self, etf_df):
        """插入ETF数据并去重"""
        if etf_df.empty:
            return 0
            
        # 添加插入时间
        etf_df['insert_time'] = datetime.now()
        
        # 批量插入（避免重复）
        records = etf_df.to_dict('records')
        result = self.etf_collection.insert_many(records, ordered=False)
        return len(result.inserted_ids)
    
    def load_news(self, start_date, end_date, sources=None):
        """加载指定时间范围的新闻"""
        query = {
            "date": {"$gte": start_date, "$lte": end_date}
        }
        if sources:
            query["source"] = {"$in": sources}
            
        cursor = self.news_collection.find(query)
        return pd.DataFrame(list(cursor))
    
    def load_featured_news(self, start_date, end_date, sources=None):
        """加载精选新闻数据"""
        query = {
            "date": {"$gte": start_date, "$lte": end_date}
        }
        if sources:
            query["source"] = {"$in": sources}
            
        cursor = self.featured_news_collection.find(query)
        return pd.DataFrame(list(cursor))
    
    def load_etf_data(self, start_date=None, end_date=None, code=None):
        """加载指定时间范围的ETF数据"""
        query = {}
        if start_date and end_date:
            query["date"] = {"$gte": start_date, "$lte": end_date}
        if code:
            query["code"] = code
            
        cursor = self.etf_collection.find(query)
        return pd.DataFrame(list(cursor))