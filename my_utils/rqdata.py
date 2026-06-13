import datetime
import json
import pandas as pd
import requests
import gzip
import io
import rqdatac as rq

import dolphindb as ddb

config = {
    'acc_db_path': 'dfs://account_years_tsdb',
    'kline_db_path': 'dfs://common_years_olap',
    'ddb_config': {
        'host': '10.140.5.44',
        'port': 8902,
        'user': 'admin',
        'password': '123456'
    }
}

# 米筐代理API
class RQData:
    def __init__(self, username='ly', password='123456'):
        self.username = username
        self.password = password

    def get_rq_data(self, func_name, param,  base_url="http://10.140.5.44:6959/"):
        """
        构建并发送HTTP POST请求，调用远程RQ数据接口

        Args:
            func_name (str): 要调用的函数名称
            param (dict): 函数的参数字典
            username (str): API认证用户名
            password (str): API认证密码
            base_url (str): API基础URL

        Returns:
            pd.DataFrame: 处理后的数据
        """
        # 构建完整URL
        url = f"{base_url.rstrip('/')}/{func_name}"

        # 构建请求体
        body = {
            "username": self.username,
            "password": self.password
        }

        # 添加参数 - 必须是字典形式
        if isinstance(param, dict):
            body.update(param)
        else:
            raise ValueError("参数必须以字典形式提供，键为参数名，值为参数值")

        try:
            # 发送POST请求
            headers = {
                "Content-Type": "application/json",
                "Accept-Encoding": "gzip",  # 支持接收压缩响应
                "Accept": "application/json"
            }

            # print(f"正在请求: {url}")
            # print(f"请求体: {body}")

            response = requests.post(
                url=url,
                data=json.dumps(body),
                headers=headers,
                timeout=300
            )

            # 检查响应状态
            if response.status_code == 401:
                print("认证失败! 请检查用户名和密码")
                raise ValueError("认证失败")

            response.raise_for_status()

            # 处理响应数据
            if response.headers.get('Content-Encoding') == 'gzip':
                # 尝试解压处理
                try:
                    # requests应该自动处理gzip，如果失败再手动处理
                    data = response.json()
                except Exception:
                    buffer = io.BytesIO(response.content)
                    with gzip.GzipFile(fileobj=buffer) as f:
                        decompressed_data = f.read().decode('utf-8')
                    data = json.loads(decompressed_data)
            else:
                data = response.json()

            # 检查是否有错误
            if isinstance(data, dict) and 'error' in data:
                raise ValueError(f"API错误: {data['error']}")

            # 处理split格式的DataFrame
            if isinstance(data, dict) and 'columns' in data and 'data' in data:
                df = pd.DataFrame(data['data'], columns=data['columns'])
                
                # 修复索引处理：检查是否有多级索引
                if 'index' in data and data['index']:
                    if isinstance(data['index'][0], list):
                        # 多级索引处理
                        index_names = ['order_book_id', 'date']  # 米筐API通常使用的索引名称
                        index_arrays = list(zip(*data['index']))
                        
                        # 将时间戳转换为日期格式（如果是时间戳）
                        if len(index_arrays) > 1:
                            # 检查第二列是否为时间戳
                            if all(isinstance(ts, (int, float)) for ts in index_arrays[1]):
                                # 转换时间戳为日期
                                date_index = pd.to_datetime(index_arrays[1], unit='ms')
                                # 重构索引数组
                                index_arrays = [index_arrays[0], date_index]
                        
                        # 创建多级索引
                        df.index = pd.MultiIndex.from_arrays(index_arrays, names=index_names)
                    else:
                        # 单级索引
                        df.index = data['index']
                
                return df

            # 其他格式的响应
            return pd.DataFrame(data)

        except Exception as e:
            print(f"请求错误: {e}")
            raise

# 米筐官方API
class RqData:
    def __init__(self):
        rq.init('license','FUqYsZ7OBunqAgtCWlyTV4FfcZr7TqYqljH47XAjrjUemHYaZAJZEZbtNbHgc9fikf92KibL6GmrPHMuplB9GiDFNjCWiOQ9jimRWb1UalPcQrA_6BzVa5EzhwMEUK4OzELB-l4xnEP_LdkzKA1ZdIBei7pmcHhFhWjYGMkiXqg=L2McHQsZ1mfYxPegD6Rzc0aWHSYy8e2rTB4MiKnHP8GmM58elDWtMoC_Jpl2ojfUFvfObmSUG1l2TYcP6VU-4fhTRkwP4kbGmXv4VOi2Nx2PREM8L92CuvhbYcQvEtVlgRfoAsjl1ZcShgFK1iq3CqWJwtt2NLEjh1ID2-wNiq0=')
        #rq.init(username='ly', password='123456')


    def get_price(self, symbol, start_date, end_date, frequency='1d', fields=None):
        return rq.get_price(symbol, start_date=start_date, end_date=end_date, frequency=frequency, fields=fields)

    
    def get_return(self, symbol, start_date, end_date, frequency='1d'):
        df = rq.get_price(symbol, start_date=start_date, end_date=end_date, frequency=frequency, fields=['prev_close', 'close'])
        df['return'] = df['close'] / df['prev_close'] - 1
        return df[['return']]
    
# 本地数据库API
class DDBData:
    def __init__(self):
        self.acc_db_path = config['acc_db_path']
        self.kline_db_path = config['kline_db_path']
        self.ddb_config = config['ddb_config']  
        #day_kline_table = 'day_kline'

    def load_data_from_dolphindb(self,db_path, table_name,columns=[],select_sql=""):
        """
        从DolphinDB加载数据,支持指定列和过滤条件
        Args:
            db_path (str): 数据库路径
            table_name (str): 表名
            columns (list): 需要加载的列名列表，默认为空表示加载所有列
            select_sql (str): 过滤条件SQL字符串，默认为空表示不使用过滤条件
        """
        # fetch_size = 1024*1024*256 # min 8192 bytes each time min value set 256M
        host = self.ddb_config['host']
        s = ddb.session()
        s.connect(host, self.ddb_config['port'], self.ddb_config['user'], self.ddb_config['password'])
        load_table_str = ''
        if s.existsTable(db_path, table_name):
            load_table_str = "loadTable(\"" + db_path + "\",\"" + table_name + "\")"
        else:
            print("Table not exist")
            return pd.DataFrame()
        load_data_str = ''
        if columns.__len__() > 0:
            load_data_str = "select " + ",".join(columns) + " from " + load_table_str
        else:
            load_data_str = "select * from " + load_table_str
        if select_sql:
            load_data_str += " where " + select_sql
        df_result = s.run(load_data_str,clearMemory=True)
        # df_result = pd.DataFrame()
        # while block.hasNext():
        #   df_result = pd.concat([df_result,block.read()])
        return df_result
    
    def write_to_dolphindb(self,df, db_path, table_name):
        """将DataFrame写入DolphinDB表，支持大数据分批写入
        Args:
            df (pd.DataFrame): 需要写入的DataFrame
            db_path (str): 数据库路径
            table_name (str): 表名
        """
        # print("Writing data to DolphinDB...")
        part_size=500   # 500M 当数据超过part_size时拆分插入
        host = self.ddb_config['host']
        s = ddb.session()
        s.connect(host, self.ddb_config['port'], self.ddb_config['user'], self.ddb_config['password'])
        ptableAppender = ddb.tableAppender(dbPath=db_path, tableName=table_name, ddbSession=s)
        is_tsdb = db_path.split('_')[0] == 'tsdb'
        df_memory_size = df.memory_usage(deep=True).sum()
        if df_memory_size >=1024*1024*1024:
            part_num = int(df.memory_usage(deep=True).sum()/1024/1024/part_size) + 1
            print("part_num",part_num)
            self.write_to_dolphindb_by_partition(df, ptableAppender,part_num)
        else:
            ret = ptableAppender.append(df)
            print(f"df rows {len(df)} , insert rows {ret}")
        s.close()

    def write_to_dolphindb_by_partition(self,df, ptableAppender,part_num):
        print(part_num)
        part_len = int(len(df)/part_num) + 1
        print(part_len)
        ret = 0
        for i in range(part_num):
            if i == part_num - 1:
                ret = ptableAppender.append(df.iloc[i*part_len:])
            else:
                ret = ptableAppender.append(df.iloc[i*part_len:(i+1)*part_len])
            print("insert rows",ret)


    # 1.封装查询快速查询函数
    # 查询资产信息
    def query_asset_info(self, date, fund_name, asset_id):
        """
        date: str,任意格式即可
        fund_name: str,基金名称
        asset_id: str,资产id,如'000001.XSHE'
        """
        # 从数据库获取行情
        date_dol = pd.to_datetime(date).strftime('%Y.%m.%d')
        quote_db_path = self.acc_db_path
        quote_table = 'fund_asset'
        sql_str = f"fund_name= '{fund_name}' and date = {date_dol} and asset_id = '{asset_id}'"
        quote_df = self.load_data_from_dolphindb(quote_db_path, quote_table, [], sql_str)
        return quote_df

    def query_day_kline(self, symbol_list, start_date, end_date):
        """
        symbol_list: list of symbol,格式如['000001.XSHE','000002.XSHE']
        start_date: str,格式如'2022-01-01'
        end_date: str,格式如'2022-01-01'
        """
        #trade_date = get_trade_date_list(start_date,end_date)
        quote_db_path = self.kline_db_path
        quote_table = 'day_kline'
        sql_str = f"order_book_id in {symbol_list} and date >= {pd.to_datetime(start_date).strftime('%Y.%m.%d')} and date <= {pd.to_datetime(end_date).strftime('%Y.%m.%d')}"
        quote_df = self.load_data_from_dolphindb(quote_db_path, quote_table, [], sql_str)
        quote_df['pct'] = quote_df['close']/quote_df['prev_close'] - 1
        return quote_df

    def query_m1_kline(self, symbol_list, start_time, end_time):
        """
        symbol_list: list of symbol,格式如['000001.XSHE','000002.XSHE']
        start_time: str,格式如'2022-01-01 09:30:00'
        end_time: str,格式如'2022-01-01 15:00:00'
        """
        quote_db_path = self.kline_db_path
        quote_table = 'one_min_kline'
        sql_str = f"order_book_id in {symbol_list} and trade_time >= {pd.to_datetime(start_time).strftime('%Y.%m.%d %H:%M:%S')} and trade_time <= {pd.to_datetime(end_time).strftime('%Y.%m.%d %H:%M:%S')}"
        quote_df = self.load_data_from_dolphindb(quote_db_path, quote_table, [], sql_str)
        return quote_df
            
    def get_price(self, symbol, start_date, end_date, fields=['close'], frequency='1d'):
        if frequency == '1d':
            return self.query_day_kline(symbol, start_date, end_date)[fields]
        elif frequency == '1m':
            return self.query_m1_kline(symbol, start_date, end_date)[fields]
        else:
            raise ValueError("不支持的频率类型，仅支持'1d'和'1m'")

    # 利用查询到的1分钟行情数据计算TWAP
    def calculate_twap(self, order_book_ids, start_time, end_time):
        quote_df = self.query_m1_kline(order_book_ids, start_time, end_time)
        if quote_df.empty:
            return pd.DataFrame(columns=["order_book_id", "start_time", "end_time", "minute_count", "twap"])
        result = (
            quote_df.groupby("order_book_id", as_index=False)
            .agg(
                start_time=("trade_time", "min"),
                end_time=("trade_time", "max"),
                minute_count=("close", "size"),
                twap=("close", "mean")
            )
        )
        result = pd.DataFrame({"order_book_id": order_book_ids}).merge(
            result,
            on="order_book_id",
            how="left"
        )
        return result

class DDBIndicator(DDBData):
    def calculate_twap(self, order_book_ids, start_time, end_time):
        quote_df = self.query_m1_kline(order_book_ids, start_time, end_time)
        if quote_df.empty:
            return pd.DataFrame(columns=["order_book_id", "start_time", "end_time", "minute_count", "twap"])

        result = (
            quote_df.groupby("order_book_id", as_index=False)
            .agg(
                start_time=("trade_time", "min"),
                end_time=("trade_time", "max"),
                minute_count=("close", "size"),
                twap=("close", "mean"),
            )
        )
        return pd.DataFrame({"order_book_id": order_book_ids}).merge(result, on="order_book_id", how="left")

# 建立一个通用查询数据类,遍历三个数据源,当失败时使用另外的数据源
class Rq_Data(RqData,RQData,DDBData):
    def __init__(self):
        # 优先顺序:先使用DDBData,如果失败再使用RQData,如果RQData失败再使用RqData
        #self.Rq = RqData()
        self.RQ = RQData()
        self.DDB = DDBData()
        

    def get_price(self, symbol, start_date, end_date, fields=[], frequency='1d'):
        """
        symbol: list of symbol,格式如['000001.XSHE','000002.XSHE']
        start_date: str,格式如'2022-01-01'
        end_date: str,格式如'2022-01-01'
        fields: list of fields,格式如['close','open','high','low','volume','amount']
        frequency: str,数据频率，支持'1d'和'1m'
        """
        # 入参校验（避免无效请求）
        if not symbol:
            return pd.DataFrame()
        if isinstance(symbol, str):
            symbol = [symbol]
        
        # 定义数据源优先级，按顺序尝试，成功就返回
        data_sources = [
            ("DolphinDB", lambda: self.DDB.get_price(symbol, start_date, end_date, fields, frequency)),
            ("内部RQ代理", lambda: self.RQ.get_rq_data('get_price', {
                'order_book_ids': symbol, 'start_date': start_date, 'end_date': end_date, 
                'fields': fields, 'frequency': frequency
            })),
        ]
        
        # 依次尝试所有数据源
        last_exception = None
        for source_name, query_func in data_sources:
            try:
                df = query_func()
                if not df.empty:
                    return df
                #print(f"{source_name}查询返回空数据")
            except Exception as e:
                last_exception = e
                print(f"{source_name}查询失败: {str(e)}")
        
        # 所有数据源都失败
        print(f"所有数据源均查询失败，最后错误: {str(last_exception)}")
        return pd.DataFrame()
    
# 创建实例
if __name__ == "__main__":
    rq=Rq_Data()
    data = rq.get_price(symbol=['000001.XSHE', '000048.XSHE'], start_date='2026-03-23', end_date='2026-03-23', frequency='1d')
    rqdata= RQData(username='ly',password='123456')
    data1 = rqdata.get_rq_data('get_price',
                              {'start_date': '2026-03-23',
                               'end_date': '2026-03-23',
                               'frequency': '1d',
                                'order_book_ids': ['000001.XSHE', '000048.XSHE']
                              }
    )
    data12 = rqdata.get_rq_data('get_vwap',
                              {'start_date': '2026-03-23',
                               'end_date': '2026-03-23',
                               'frequency': '5m',
                                'order_book_ids': ['000001.XSHE', '000048.XSHE']
                              }
    )
    print(data1)
    print(data12)

    indicator = DDBIndicator()
    twap_result = indicator.calculate_twap(order_book_ids=['000001.XSHE', '000048.XSHE'], start_time='2025-01-02 09:30:00', end_time='2025-01-02 10:30:00')
    print(twap_result)
