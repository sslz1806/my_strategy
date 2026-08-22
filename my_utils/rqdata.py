from __future__ import annotations

import datetime
import json
import logging
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

# 米筐 ETF 行情只请求落盘所需字段，减少单次响应体积。
ETF_DAY_FIELDS = [
    'open',
    'high',
    'low',
    'close',
    'prev_close',
    'volume',
    'total_turnover',
]
ETF_MINUTE_FIELDS = [
    'open',
    'high',
    'low',
    'close',
    'volume',
    'total_turnover',
]


# 米筐官方API
class RqData: #NhgWOWVZCnNzlhXcU7QwoYWSxsDredHhgzOprOOpZ2SbpVJGFm9b1W4fRs61v-cYaAEI3RT9_UoPmkmC1P_kGFoDJ1TMRqQmvXEj0RPQpBlGVXF8r8blbvLe5iqLwUe0yXGAxA-5ET5YfEPnQFYGpufhbzDEokf3Tb319vhwvB4=R9Wl1ULCiS05KJVkb7LM8Vbj_H1TwVnbiXhGKJBszyPtawrOWxlcCKwKQAu0z22uiSAwJjR_1Zdqkw4hf-762Eo46rPKJU3J7TfivcWdbPk6-QvygSzQ7NtDnx6k-jn1tPOHQFljsE6mMV4ARhX_otU7A2MicLXwwgSxiLTrKmk=
    def __init__(self):
        rq.init('license','NhgWOWVZCnNzlhXcU7QwoYWSxsDredHhgzOprOOpZ2SbpVJGFm9b1W4fRs61v-cYaAEI3RT9_UoPmkmC1P_kGFoDJ1TMRqQmvXEj0RPQpBlGVXF8r8blbvLe5iqLwUe0yXGAxA-5ET5YfEPnQFYGpufhbzDEokf3Tb319vhwvB4=R9Wl1ULCiS05KJVkb7LM8Vbj_H1TwVnbiXhGKJBszyPtawrOWxlcCKwKQAu0z22uiSAwJjR_1Zdqkw4hf-762Eo46rPKJU3J7TfivcWdbPk6-QvygSzQ7NtDnx6k-jn1tPOHQFljsE6mMV4ARhX_otU7A2MicLXwwgSxiLTrKmk=')
        #rq.init(username='ly', password='123456')

    def close(self):
        """释放本进程的米筐官方连接，供长驻或复用 main 的调用方在结束后收尾。"""
        rq.reset()


    def get_price(
        self,
        symbol,
        start_date,
        end_date,
        frequency='1d',
        fields=None,
        adjust_type='pre',
        skip_suspended=False,
        expect_df=True,
    ):
        """调用官方行情接口，并保持既有调用默认使用前复权。"""
        return rq.get_price(
            symbol,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            fields=fields,
            adjust_type=adjust_type,
            skip_suspended=skip_suspended,
            expect_df=expect_df,
        )

    def get_etf_instruments(self):
        """获取包含退市记录的完整 ETF 历史基础池。"""
        return rq.all_instruments(type='ETF', market='cn')

    def get_trading_days(self, start_date, end_date):
        """获取一次运行内可复用的中国市场交易日历。"""
        values = rq.get_trading_dates(start_date, end_date, market='cn')
        return [value.date() if hasattr(value, 'date') else value for value in values]

    def get_quota(self):
        """读取官方账户当日流量上限、已用量、许可类型及剩余有效期。"""
        return rq.user.get_quota()

    def fetch_etf_day_range(self, rq_codes, start_date, end_date):
        """用一次请求获取批次内全部 ETF 的不复权日线。"""
        return self.get_price(
            rq_codes,
            start_date,
            end_date,
            frequency='1d',
            fields=ETF_DAY_FIELDS,
            adjust_type='none',
            skip_suspended=False,
            expect_df=True,
        )

    def fetch_etf_minute_range(self, rq_codes, start_date, end_date):
        """用一次请求获取批次内全部 ETF 的原始不复权 1 分钟线。"""
        return self.get_price(
            rq_codes,
            start_date,
            end_date,
            frequency='1m',
            fields=ETF_MINUTE_FIELDS,
            adjust_type='none',
            skip_suspended=False,
            expect_df=True,
        )

    
    def get_return(self, symbol, start_date, end_date, frequency='1d'):
        df = rq.get_price(symbol, start_date=start_date, end_date=end_date, frequency=frequency, fields=['prev_close', 'close'])
        df['return'] = df['close'] / df['prev_close'] - 1
        return df[['return']]
    
# 本地数据库API
class DDBData:
    def __init__(self, session=None):
        self.acc_db_path = config['acc_db_path']
        self.kline_db_path = config['kline_db_path']
        self.ddb_config = config['ddb_config']
        self._session = session
        self._owns_session = session is None
        #day_kline_table = 'day_kline'

    def connect(self):
        """返回可复用的 DDB 会话；未注入会话时按现有配置延迟创建。"""
        if self._session is None:
            self._session = ddb.session()
            self._session.connect(
                self.ddb_config['host'],
                self.ddb_config['port'],
                self.ddb_config['user'],
                self.ddb_config['password'],
            )
        return self._session

    def close(self):
        """关闭本实例创建的会话，不接管调用方注入会话的生命周期。"""
        if self._session is not None and self._owns_session:
            self._session.close()
            self._session = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def _run_query(self, script, query_name):
        """在数据访问层统一补充查询上下文，并保留原始异常链。"""
        try:
            return self.connect().run(script)
        except Exception as exc:
            logging.warning("DDB %s查询失败: %s", query_name, exc)
            raise RuntimeError(f"DDB {query_name}查询失败: {exc}") from exc

    def get_stock_universe(self):
        """获取 DDB 中的米筐普通股票代码列表。"""
        try:
            data = self.connect().run("""
                select order_book_id
                from loadTable('dfs://common_years_tsdb', 'instrument_base')
                where type = 'CS'
            """)
            return data["order_book_id"].dropna().tolist()
        except Exception as exc:
            logging.error("获取 DDB 股票池失败: %s", exc)
            return []

    def get_trading_days(self, start_date, end_date):
        """从 DDB 交易日表读取指定范围内的交易日。"""
        start_str = pd.to_datetime(start_date).strftime("%Y.%m.%d")
        end_str = pd.to_datetime(end_date).strftime("%Y.%m.%d")
        try:
            data = self.connect().run(f"""
                select distinct trade_date
                from loadTable('dfs://common_years_tsdb', 'trade_date')
                where is_trade_date = true
                  and trade_date >= date({start_str})
                  and trade_date <= date({end_str})
                order by trade_date
            """)
            dates = data["trade_date"].dropna().tolist()
            return [value.date() if hasattr(value, "date") else value for value in dates]
        except Exception as exc:
            logging.error("获取交易日列表失败: %s", exc)
            return []

    def fetch_day_range(
        self,
        start_date,
        end_date,
        rq_codes=None,
        allowed_dates=None,
    ):
        """批量查询 DDB 日线相关表并返回本地统一 Schema 的 Polars 数据。"""
        from my_utils.rq_fun import RQ_DAY_SCHEMA, normalize_ddb_day_range

        if rq_codes is None:
            rq_codes = self.get_stock_universe()
        if not rq_codes:
            import polars as pl

            return pl.DataFrame(schema=RQ_DAY_SCHEMA)

        start_str = pd.to_datetime(start_date).strftime("%Y.%m.%d")
        end_str = pd.to_datetime(end_date).strftime("%Y.%m.%d")
        kline = self._run_query(f"""
            select order_book_id, date as trading_date,
                   open, close, high, low,
                   volume, total_turnover as amount,
                   prev_close as pre_close,
                   limit_up, limit_down
            from loadTable('dfs://common_years_olap', 'day_kline')
            where (order_book_id like '%.XSHE' or order_book_id like '%.XSHG')
              and date >= date({start_str})
              and date <= date({end_str})
        """, "日线")
        if kline.empty:
            import polars as pl

            return pl.DataFrame(schema=RQ_DAY_SCHEMA)

        is_st = self._run_query(f"""
            select order_book_id, date as trading_date, is_st
            from loadTable('dfs://stock_years_tsdb', 'is_st_stock')
            where date >= date({start_str})
              and date <= date({end_str})
        """, "ST 标记")
        shares = self._run_query(f"""
            select order_book_id, date as trading_date,
                   circulation_a, total_a, free_circulation
            from loadTable('dfs://stock_years_tsdb', 'stock_shares')
            where date >= date({start_str})
              and date <= date({end_str})
        """, "股本")
        ex_factor = self._run_query(f"""
            select order_book_id, ex_date, ex_cum_factor as adj_factor
            from loadTable('dfs://stock_years_tsdb', 'ex_factor')
            where ex_date <= date({end_str})
        """, "复权因子")
        instruments = self._run_query("""
            select order_book_id, symbol as name
            from loadTable('dfs://common_years_tsdb', 'instrument_base')
            where type = 'CS'
        """, "股票信息")

        return normalize_ddb_day_range(
            kline,
            is_st,
            shares,
            ex_factor,
            instruments,
            rq_codes,
            allowed_dates=allowed_dates,
        )

    def fetch_minute_range(
        self,
        start_date,
        end_date,
        allowed_dates=None,
        rq_codes=None,
    ):
        """批量查询 DDB 一分钟线并合成为右对齐 15 分钟线。"""
        import polars as pl
        from my_utils.rq_fun import RQ_MIN_SCHEMA, aggregate_right_aligned_15min

        if rq_codes is None:
            rq_codes = self.get_stock_universe()
        if not rq_codes:
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

        if allowed_dates is None:
            allowed_dates = set(self.get_trading_days(start_date, end_date))
        else:
            allowed_dates = set(allowed_dates)
        if not allowed_dates:
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)

        code_filter = ""
        if len(rq_codes) == 1:
            safe_code = str(rq_codes[0]).replace("'", "''")
            code_filter = f"\n              and order_book_id = '{safe_code}'"

        start_str = pd.to_datetime(start_date).strftime("%Y.%m.%d")
        end_str = pd.to_datetime(end_date).strftime("%Y.%m.%d")
        raw = self._run_query(f"""
            select order_book_id, trade_time,
                   open, close, high, low, volume, total_turnover
            from loadTable('dfs://common_years_olap', 'one_min_kline')
            where (order_book_id like '%.XSHE' or order_book_id like '%.XSHG')
              and trade_time >= timestamp({start_str} 09:31:00)
              and trade_time <= timestamp({end_str} 15:00:00)
              {code_filter}
        """, "分钟线")

        if raw.empty:
            return pl.DataFrame(schema=RQ_MIN_SCHEMA)
        return aggregate_right_aligned_15min(
            pl.from_pandas(raw),
            rq_codes=rq_codes,
            allowed_dates=allowed_dates,
        )

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
