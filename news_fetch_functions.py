from akshare.request import make_request_with_retry_json
import pandas as pd
import time

def news_cx_kx(total_size=0, batch_size=200, last_date=None, earliest_date=None) -> pd.DataFrame:
    """
    财新网-一线快讯
    """
    url = "https://cxdata.caixin.com/api/dataplus/kxNews"
    PAGE_SIZE = 25  # 每页数据条数固定为25条
    
    # 从page_now开始直到获取到batch_size条数据或无更多数据
    page_now = total_size // PAGE_SIZE + 1 if total_size > 0 else 1
    all_df = pd.DataFrame(columns=["title", "content", "date"])
    
    while True:
        params = {
            "pageNum": f"{page_now}",
            "pageSize": f"{PAGE_SIZE}",
            "showLabels": "true",
        }
        data_json = make_request_with_retry_json(url, params=params)
        temp_df = pd.DataFrame(data_json["data"])
        print(f"--- Fetching page {page_now} with {temp_df.shape[0]} records...")
        # 返回空时处理
        if temp_df.empty:
            print(f"--- No more data available from page {page_now}. Stopping fetch.")
            break

        temp_df = temp_df[["title", "text", "date", "time"]]
        temp_df.columns = ["title", "content", "date", "time"]
        temp_df['title'] = temp_df['title'].fillna('')
        # 'content' 删除换行符
        temp_df['content'] = temp_df['content'].str.replace('\r\n', '', regex=False)
        temp_df["date"] = pd.to_datetime(
            temp_df["date"] + ' ' + temp_df['time'], errors="coerce", format='%Y/%m/%d %H:%M'
        )

        # 增量过滤
        if last_date and earliest_date:
            temp_df = temp_df[(temp_df['date'] > last_date) | (temp_df['date'] < earliest_date)]
        elif last_date:
            temp_df = temp_df[temp_df['date'] > last_date]
        elif earliest_date:
            temp_df = temp_df[temp_df['date'] < earliest_date]
        if temp_df.empty:
            print(f"--- No new data after filtering. Stopping fetch.")
            break
        temp_df = temp_df[["title", "content", "date"]]  # Select only required columns
        all_df = pd.concat([all_df, temp_df], ignore_index=True) if not all_df.empty else temp_df

        # 检查是否已获取到足够batch_size的数据
        if all_df.shape[0] >= batch_size:
            all_df = all_df.head(batch_size)
            break

        time.sleep(1)  # 避免请求过快
        page_now += 1
    
    return all_df[["title", "content", "date"]]

# {source_name: function_name}
function_dict = {
    "财新网": news_cx_kx,
}