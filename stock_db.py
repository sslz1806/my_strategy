"""
股票数据库api,包括插入,更新,查询等功能
"""
import pandas as pd
import polars as pl
import pymysql
from pymysql.converters import conversions
from stock_api import retry_with_timeout
from dbutils.pooled_db import PooledDB
import logging
from mapping import *
db_config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'root',
    'database': 'stock_db',
}
db_pool =PooledDB(
    creator=pymysql,
    maxconnections=20,
    ping=0,
    **db_config,
    charset='utf8mb4'
)
class MSDB:
    # mysql数据库api
    """
    增:增加日线数据,分钟数据,等。支持多线程增量写入同一个表格或者不同表格。
    更:覆盖更新。支持多线程更新
    查:按照交易日,股票代码等查询。支持多线程查询同一个表格或不同表格。
    """
    def __init__(self,db_config=db_config,maxconn=20):
        self.db_config = db_config
        self.db_name = db_config['database']
        self.db_pool = PooledDB(
            creator=pymysql,
            maxconnections=maxconn,
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database'],
            charset='utf8mb4'
        )
        # 1. 自定义转换器：把 MySQL 的 decimal 直接转为 float（读出来就是数值类型）
        def decimal_to_float(val):
            return float(val) if val is not None else None
        
        # 2. 替换默认转换规则
        custom_conv = conversions.copy()
        custom_conv[pymysql.FIELD_TYPE.NEWDECIMAL] = decimal_to_float  # decimal 转 float
        custom_conv[pymysql.FIELD_TYPE.DECIMAL] = decimal_to_float
        # 添加日志功能
        self.logger = logging.getLogger('MSDB')
        self.logger.setLevel(logging.INFO)
        # 关键：判断日志器是否已有处理器（避免主程序有配置时重复添加）
        if not self.logger.handlers:
            # 1. 创建“控制台处理器”（兜底输出，主程序无配置时至少能在控制台看到日志）
            console_handler = logging.StreamHandler()
            # 2. 创建日志格式器（包含关键信息，便于调试）
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s [%(levelname)s] - %(message)s'  # 格式：时间 - 日志器名 - 级别 - 内容
            )
            # 3. 绑定格式器到处理器
            console_handler.setFormatter(formatter)
            # 4. 绑定处理器到日志器
            self.logger.addHandler(console_handler)
            
            # （可选）添加“文件处理器”，即使主程序无配置，也会自动写入日志文件
            file_handler = logging.FileHandler(
                filename='msdb.log',  # 日志文件路径，可自定义
                encoding='utf-8',     # 避免中文乱码
                mode='a'              # 追加模式，避免覆盖历史日志
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)


    def update_stocks_data(self,df,tb_name='ts_stock_all_data',mode='update',batch_size=5000):
        """
        对tb_name中的表更新数据
        说明:表中以trading_date,code为唯一索引。mode=ignore或insert表示忽略重复插入,mode=update表示更新
        :param tb_name: 表名,默认为ts_stock_all_data。以trading_date,code为唯一索引。插入时按照trading_date,code排序。
        :param df: polars.DataFrame格式的数据,以trading_date,code为主键
        :param mode: 插入模式,默认为ignore。可选值有'ignore','update'。分别表示忽略重复数据,替换重复数据,更新重复数据。
        """
        conn = self.db_pool.connection()
        cursor = conn.cursor()
        # 1.处理列名
        # 获取表的列名
        cursor.execute(f"SHOW COLUMNS FROM {tb_name}")
        columns = [column[0] for column in cursor.fetchall()]
        # 获取数据的列名
        df_columns = df.columns
        # 对比数据的列名和表的列名
        common_columns = [col for col in df_columns if col in columns]
        quoted_columns = [f"`{col}`" for col in common_columns]  # 添加反引号以防列名为SQL关键字
        if common_columns==[]:
            self.logger.error(f"数据的列名和表{tb_name}的列名没有交集,无法插入数据")
            raise ValueError(f"数据的列名和表{tb_name}的列名没有交集,无法插入数据")
        # 打印不在表中的列名
        missing_columns = [col for col in df_columns if col not in columns]
        self.logger.info(f"以下列名不在表{tb_name}中,将被忽略: {missing_columns}")
        
        # 2.构造更新语句
        if mode == 'ignore' or mode=='insert':
            insert_sql = f"INSERT IGNORE INTO `{tb_name}` ({', '.join(quoted_columns)}) VALUES ({', '.join(['%s'] * len(quoted_columns))})"
        elif mode == 'update' or mode=='replace':
            update_clause = ", ".join([f"{col} = VALUES({col})" for col in quoted_columns if col not in ['code', 'trading_date']])
            insert_sql = f"INSERT INTO `{tb_name}` ({', '.join(quoted_columns)}) VALUES ({', '.join(['%s'] * len(quoted_columns))}) ON DUPLICATE KEY UPDATE {update_clause}"
        else:
            raise ValueError("mode参数只能为'ignore'或'replace'")
        
        # 3.插入数据
        try:
            sub_df = df.select(common_columns)
            row_iter = sub_df.iter_rows()  # 返回 tuple 的生成器（polars 高效）

            # 4) 分批写入
            conn.autocommit = False
            batch = []
            total = 0
            for row in row_iter:
                batch.append(row)
                if len(batch) >= batch_size:
                    cursor.executemany(insert_sql, batch)
                    total += len(batch)
                    batch.clear()
            if batch:
                cursor.executemany(insert_sql, batch)
                total += len(batch)

            conn.commit()
            self.logger.info(f"{tb_name} 写入完成，共 {total} 行，模式 {mode}，批量 {batch_size}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"插入数据失败: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

    def query_data(self,sql):
        """
        查询数据
        :param sql: 查询语句
        :return: polars.DataFrame格式的数据
        """
        conn = self.db_pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pl.DataFrame(result, columns=columns)
            return df
        except Exception as e:
            self.logger.info(f"查询数据失败: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()

    def get_stocks_data(self,stock_list=None,start_date='2025-11-01',end_date='2025-11-11',tb_name='ts_stock_all_data'):
        """
        获取多个股票在某个时间段内的日线数据
        :param stock_list: 股票代码列表,默认为None,表示获取所有股票的数据
        :param start_date: 开始日期,格式默认为'YYYY-MM-DD',其他格式也可以
        :param end_date: 结束日期,格式默认为'YYYY-MM-DD'
        :param tb_name: 表名,默认为ts_stock_all_data
        :return: polars.DataFrame格式的数据
        """

        start_date = convert_date_format(start_date,to_format='date')
        end_date = convert_date_format(end_date,to_format='date')
        conn = self.db_pool.connection()
        cursor = conn.cursor()
        try:
            if stock_list is None:
                sql = f"SELECT * FROM {tb_name} WHERE trading_date BETWEEN '{start_date}' AND '{end_date}' ORDER BY trading_date, code"
            else:
                stock_str = ','.join([f"'{code}'" for code in stock_list])
                sql = f"SELECT * FROM {tb_name} WHERE code IN ({stock_str}) AND trading_date BETWEEN '{start_date}' AND '{end_date}' ORDER BY trading_date, code"
            cursor.execute(sql)
            result = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pl.DataFrame(result,schema=columns)
            self.logger.info(f"成功获取股票数据: {len(df)}条记录")
            return df
        except Exception as e:
            self.logger.error(f"获取股票数据失败: {e}")
            raise e
        finally:
            cursor.close()
            conn.close()


if __name__ == "__main__":
    from fun import *
    db = MSDB()
    #df = read_day_data(start_date='2021-11-11',end_date='2022-11-11')
    #db.update_stocks_data(df,mode='update')
    data = db.get_stocks_data(start_date='2021-01-01',end_date='2022-11-10')
    print(data)