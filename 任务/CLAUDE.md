# 任务目录 - 数据更新行为规范

> 本规范用于指导AI在进行股票数据更新时的行为准则

## 零、模块职责与复用硬约束

数据更新脚本是生产入口，应保持为“薄编排层”，不得把可复用的数据访问或清洗逻辑重新写回主脚本。

- 开发新逻辑前，先搜索 `my_utils/` 是否已有同类函数或类；已有接口能力不足时，优先增加向后兼容的可选参数或输入类型，不新增同义平行函数。
- 数据源连接、认证、请求和原始查询放在对应 API 模块，例如掘金/Tushare 放在 `stock_api.py`，米筐和米筐 DDB 放在 `rqdata.py`。
- 跨数据源通用的股票代码、日期和列名转换放在 `mapping.py`，并优先扩展 `convert_code_format()` 等现有接口。
- 数据源专属的字段合并、Schema 对齐、分钟聚合和质量校验放在对应工具模块，例如米筐逻辑放在 `rq_fun.py`。
- `任务/` 下的主脚本只保留 CLI 参数、批次循环、调用顺序、日志、重试和退出码；不得直接重复 SQL、代码转换、Schema 对齐或分区存储实现。
- 只有新能力与现有接口职责明显不同、强行扩展会破坏兼容性时，才允许新增公共接口，并在实现说明中写明原因。

## 一、数据源配置

### 1.1 Tushare数据源
- 主token: `ts_token = 'YzAEH11Yc7jZCHjeJa63fnbpSt3k9Je3GvWn0390oiBKO95bVJjP7u5L34e2ff6b'`
- 分钟token: `mins_token = 'fbdsJ45z9Nodp7FbUgDEsm1Oi8boH7Wuiqn7cQJnRAvs5bSwuB4e0iOBbe16ef40'`

### 1.2 掘金数据源
- token: `gm_token` (在stock_api.py中配置)

## 二、数据存储路径

| 数据类型 | 存储路径 | 数据源 |
|----------|----------|--------|
| 日线行情 | `E:\working\stock_data\ts_stock_all_data` | tushare |
| 复权因子 | `E:\working\stock_data\ts_adj` | tushare |
| 每日指标 | `E:\working\stock_data\ts_daily_basic` | tushare |
| 15分钟数据 | `E:\working\stock_data\15min_stock_data_dir` | 掘金 |

## 三、核心流程：数据调用检查与清洗对齐

### 3.1 标准更新流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     数据更新标准流程                              │
├─────────────────────────────────────────────────────────────────┤
│  1.【数据源调用】                                                │
│     ├── API连通性检查                                            │
│     ├── 获取数据 + 初步验证                                      │
│     └── 异常处理（重试/降级/报警）                                │
│                          ↓                                      │
│  2.【数据清洗】                                                  │
│     ├── 列名映射（数据源 → 本地格式）                            │
│     ├── 股票代码格式转换                                         │
│     ├── 数据类型转换                                             │
│     └── 缺失值/异常值处理                                        │
│                          ↓                                      │
│  3.【数据对齐】                                                  │
│     ├── Schema对齐（强制转换）                                   │
│     ├── 日期格式统一                                             │
│     └── 与本地数据对比校验                                       │
│                          ↓                                      │
│  4.【增量合并】                                                  │
│     ├── 读取本地已有数据                                         │
│     ├── 找出新增/更新的记录                                      │
│     ├── 合并去重                                                 │
│     └── 写入存储                                                 │
│                          ↓                                      │
│  5.【完整性校验】                                                │
│     ├── 记录数对比                                              │
│     ├── 日期范围验证                                            │
│     └── 关键字段检查                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 第一步：数据源调用检查

#### 3.2.1 API连通性检查
```python
def check_api_connection(api_name="tushare"):
    """检查API是否可用"""
    try:
        if api_name == "tushare":
            api = stock_api()
            test_data = api.ts_get_daily_basic('2024-01-01', '2024-01-01')
            return test_data is not None and not test_data.empty
        elif api_name == "gm":
            import gm.api as gm
            gm.set_token(GM_TOKEN)
            data = gm.history(symbol='SZSE.000001', start_time='2024-01-01', end_time='2024-01-01')
            return data is not None
    except Exception as e:
        logging.error(f"{api_name} API连接失败: {e}")
        return False
```

#### 3.2.2 数据拉取与初步验证
```python
def fetch_data_with_validation(api_func, *args, **kwargs):
    """
    带验证的数据拉取
    返回: (成功标志, 数据或错误信息)
    """
    try:
        data = api_func(*args, **kwargs)

        # 初步验证1: 数据非空
        if data is None or data.empty:
            return False, "API返回空数据"

        # 初步验证2: 必要字段存在
        required_fields = kwargs.get('required_fields', [])
        if required_fields:
            missing = [f for f in required_fields if f not in data.columns]
            if missing:
                return False, f"缺少必要字段: {missing}"

        # 初步验证3: 数据量级合理
        if len(data) < kwargs.get('min_rows', 1):
            return False, f"数据行数异常: {len(data)}"

        return True, data

    except Exception as e:
        return False, f"API调用异常: {str(e)}"
```

#### 3.2.3 异常处理策略
| 异常类型 | 处理策略 |
|----------|----------|
| 网络超时 | 重试3次，每次间隔5秒 |
| API限流 | 等待1分钟，加入队列 |
| 认证失败 | 立即终止，发送警报 |
| 数据异常 | 记录日志，跳过该批次 |

### 3.3 第二步：数据清洗

#### 3.3.1 列名映射（核心！）

**Tushare → 本地格式**
```python
# Tushare返回的列名
TS_COLUMNS = {
    'trade_date': 'trading_date',    # 日期列名统一
    'ts_code': 'code',               # 股票代码
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'vol': 'volume',                 # 成交量列名统一
}

def map_column_names(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """列名映射"""
    return df.rename(columns=column_map)
```

**掘金 → 本地格式**
```python
# 掘金返回的列名
GM_COLUMNS = {
    'symbol': 'code',                # 股票代码
    'time': 'datetime',              # 时间列
    'volume': 'volume',
}
```

#### 3.3.2 股票代码格式转换
```python
from my_utils.mapping import convert_code_format

def standardize_stock_code(code: str, to_format: str = 'gm') -> str:
    """
    股票代码格式转换

    格式说明:
    - tushare: '000001.XSHE' (后缀XSHE=深交所, XSHG=上交所)
    - gm格式: 'SZSE.000001' (前缀SZSE=深交所, SHSE=上交所)

    示例:
    - '000001.XSHE' → 'SZSE.000001'
    - '600000.XSHG' → 'SHSE.600000'
    """
    return convert_code_format(code, format=to_format)

def batch_standardize_codes(codes: list, to_format: str = 'gm') -> list:
    """批量转换股票代码"""
    return [standardize_stock_code(c, to_format) for c in codes]
```

#### 3.3.3 数据类型转换
```python
def cast_data_types(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    """
    强制数据类型转换

    常见转换:
    - date/datetime: 确保日期格式一致
    - float: 确保精度
    - string: 确保无前导零问题
    """
    for col, dtype in schema.items():
        if col in df.columns:
            try:
                df = df.with_columns(pl.col(col).cast(dtype).alias(col))
            except Exception as e:
                logging.warning(f"列{col}类型转换失败: {e}")
    return df
```

#### 3.3.4 缺失值与异常值处理
```python
def clean_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    数据清洗

    处理策略:
    - 缺失值: 跳过含缺失值的行（对于行情数据）
    - 涨停价异常: limit_up <= 0 → 标记为无效
    - 价格异常: close <= 0 或 close > 10000 → 标记为无效
    - 日期异常: trading_date > today → 过滤
    """
    original_count = len(df)

    # 1. 删除关键字段为空的行
    key_fields = ['code', 'trading_date', 'open', 'high', 'low', 'close']
    for field in key_fields:
        if field in df.columns:
            df = df.filter(pl.col(field).is_not_null())

    # 2. 价格合理性检查
    if 'close' in df.columns:
        df = df.filter((pl.col('close') > 0) & (pl.col('close') < 100000))

    # 3. 日期范围检查
    if 'trading_date' in df.columns:
        today = datetime.date.today()
        df = df.filter(pl.col('trading_date') <= today)

    cleaned_count = len(df)
    if original_count != cleaned_count:
        logging.info(f"清洗数据: {original_count} → {cleaned_count} (删除{original_count - cleaned_count}条)")

    return df
```

### 3.4 第三步：数据对齐

#### 3.4.1 Schema对齐
```python
def align_schema(new_data: pl.DataFrame, existing_schema: dict) -> pl.DataFrame:
    """
    新数据Schema对齐到已有Schema

    原则:
    1. 类型一致: 强制转换新数据列类型匹配已有schema
    2. 列补齐: 缺失的列添加空值
    3. 列裁剪: 额外的列保留但不写入（警告）
    """
    # 类型转换
    convert_exprs = []
    for col, dtype in existing_schema.items():
        if col in new_data.columns:
            try:
                convert_exprs.append(pl.col(col).cast(dtype).alias(col))
            except:
                pass  # 转换失败保留原类型

    if convert_exprs:
        new_data = new_data.select(convert_exprs)

    # 补齐缺失列
    for col, dtype in existing_schema.items():
        if col not in new_data.columns:
            new_data = new_data.with_columns(
                pl.lit(None).cast(dtype).alias(col)
            )

    # 检查额外列
    extra_cols = [c for c in new_data.columns if c not in existing_schema.keys()]
    if extra_cols:
        logging.warning(f"新数据包含额外列（将被忽略）: {extra_cols}")

    return new_data
```

#### 3.4.2 日期格式统一
```python
def normalize_date_format(df: pl.DataFrame, date_col: str = 'trading_date') -> pl.DataFrame:
    """
    日期格式统一为 datetime.date 对象
    """
    if date_col in df.columns:
        if df[date_col].dtype == pl.String:
            df = df.with_columns(
                pl.col(date_col).str.to_date('%Y-%m-%d').alias(date_col)
            )
        elif df[date_col].dtype == pl.Datetime:
            df = df.with_columns(
                pl.col(date_col).dt.date().alias(date_col)
            )
    return df
```

### 3.5 第四步：增量合并

#### 3.5.1 确定增量范围
```python
def get_incremental_range(save_dir: str, new_data: pl.DataFrame) -> tuple:
    """
    确定增量更新范围

    返回: (待插入数据, 已有数据检查结果)
    """
    existing_dates = get_existing_dates(save_dir)  # 从目录获取已有日期

    if not existing_dates:
        return new_data, "首次全量导入"

    max_existing_date = max(existing_dates)

    # 筛选新增数据
    new_records = new_data.filter(
        pl.col('trading_date') > max_existing_date
    )

    # 检查是否有数据需要更新（同一日期的不同版本）
    updated_records = new_data.filter(
        pl.col('trading_date') <= max_existing_date
    )

    if updated_records.height > 0:
        logging.warning(f"发现{updated_records.height}条需更新的历史记录")

    return new_records, f"增量: {new_records.height}条, 更新: {updated_records.height}条"
```

#### 3.5.2 三种更新模式

```python
mode = 'insert'   # 增量更新：仅插入新日期的数据
mode = 'update'   # 全量更新：覆盖所有数据
mode = 'correct'  # 修正更新：修正指定日期的历史数据
```

#### 3.5.3 修正模式逻辑 (mode='correct')

**使用场景**：
- 发现历史数据有误需要修正
- 数据源回补了缺失的历史数据
- 复权因子等字段的历史值发生变化

**核心逻辑**：
```python
def correct_data(new_data: pl.DataFrame, save_dir: str, correct_dates: list = None):
    """
    修正模式：替换指定日期的数据

    参数:
        new_data: 新数据（含需修正的日期）
        save_dir: 存储目录
        correct_dates: 需要修正的日期列表，None表示自动判断

    处理流程:
    1. 备份旧数据
    2. 找出需要修正的日期
    3. 删除旧分区
    4. 写入新数据
    5. 记录修正日志
    """
    import shutil
    from datetime import datetime as dt

    # 1. 备份
    backup_dir = f"{save_dir}_backup_{dt.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(save_dir, backup_dir)
    logging.info(f"已备份旧数据到: {backup_dir}")

    # 2. 确定需要修正的日期
    if correct_dates is None:
        # 自动判断：取新数据和旧数据的日期交集
        new_dates = set(new_data.select('trading_date').unique().to_series())
        old_dates = set(get_existing_dates(save_dir))
        correct_dates = list(new_dates & old_dates)

    logging.info(f"需要修正的日期: {correct_dates}")

    # 3. 删除旧分区 + 4. 写入新数据
    for date in correct_dates:
        date_str = date.strftime('%Y-%m-%d')
        old_partition = os.path.join(save_dir, f'trading_date={date_str}')

        if os.path.exists(old_partition):
            shutil.rmtree(old_partition)
            logging.info(f"已删除旧分区: {old_partition}")

        # 写入该日期的新数据
        date_data = new_data.filter(pl.col('trading_date') == date)
        if date_data.height > 0:
            date_data.write_parquet(save_dir, partition_by=['trading_date'])
            logging.info(f"已写入修正数据: {date_str}, {date_data.height}条")

    # 5. 记录修正日志
    logging.info(f"修正完成，共修正{len(correct_dates)}个日期")
    return correct_dates
```

#### 3.5.4 合并去重
```python
def merge_incremental(new_data: pl.DataFrame, save_dir: str, key_cols: list = ['trading_date', 'code']) -> pl.DataFrame:
    """
    增量合并

    逻辑:
    1. 读取已有数据
    2. 与新数据按key去重合并
    3. 返回完整数据集
    """
    # 1. 读取已有数据
    existing_data = read_existing_data(save_dir)

    if existing_data is None or existing_data.height == 0:
        return new_data

    # 2. 合并去重（新数据优先）
    combined = pl.concat([new_data, existing_data])
    combined = combined.unique(subset=key_cols, keep='first', maintain_order=True)

    return combined
```

### 3.6 第五步：完整性校验

#### 3.6.1 写入前校验
```python
def pre_write_validation(new_data: pl.DataFrame, existing_data: pl.DataFrame = None) -> bool:
    """
    写入前校验

    检查项:
    1. 数据量级: 新增记录数是否合理
    2. 日期连续性: 是否有异常跳跃
    3. 字段完整性: 关键字段是否有大量缺失
    """
    checks_passed = True

    # 检查1: 数据量级（单日股票数量约4000-5000）
    if new_data.height > 10000:
        logging.warning(f"单次数据量过大: {new_data.height}条，请确认是否异常")

    # 检查2: 关键字段缺失率
    critical_fields = ['code', 'trading_date', 'close']
    for field in critical_fields:
        if field in new_data.columns:
            missing_rate = new_data.filter(pl.col(field).is_null()).height / new_data.height
            if missing_rate > 0.1:
                logging.warning(f"字段{field}缺失率过高: {missing_rate:.2%}")
                checks_passed = False

    # 检查3: 与历史数据对比（如果有时）
    if existing_data is not None and existing_data.height > 0:
        avg_daily_count = existing_data.height / existing_data.select('trading_date').n_unique()
        new_avg_count = new_data.height / new_data.select('trading_date').n_unique()
        if abs(new_avg_count - avg_daily_count) / avg_daily_count > 0.5:
            logging.warning(f"日均数据量异常: 历史{avg_daily_count:.0f} vs 新增{new_avg_count:.0f}")

    return checks_passed
```

#### 3.6.2 写入后校验
```python
def post_write_validation(save_dir: str, expected_count: int) -> bool:
    """
    写入后校验

    检查项:
    1. 记录数是否一致
    2. 最新日期是否正确
    3. 文件是否完整
    """
    try:
        # 读取最新写入的数据
        latest_data = read_latest_batch(save_dir)
        actual_count = latest_data.height

        if actual_count != expected_count:
            logging.error(f"写入后校验失败: 期望{expected_count}条，实际{actual_count}条")
            return False

        # 检查日期
        latest_date = latest_data.select('trading_date').max().item()
        if latest_date != datetime.date.today():
            logging.warning(f"最新数据日期非今日: {latest_date}")

        return True

    except Exception as e:
        logging.error(f"写入后校验异常: {e}")
        return False
```

---

## 四、新增数据源接入规范

### 4.1 接入前检查清单
- [ ] API文档阅读完毕
- [ ] 测试数据拉取成功
- [ ] 列名映射表已创建
- [ ] 代码格式已确认
- [ ] Schema对齐逻辑已验证

### 4.2 新增数据源标准模板

```python
def update_xxx_data(start_date, end_date, save_dir='ts_xxx', mode='insert'):
    """
    更新XXX数据

    【数据源】xxx API
    【数据格式】pandas/polars DataFrame
    【存储路径】E:\\working\\stock_data\\ts_xxx
    【分区方式】trading_date
    """
    # ==================== 1. 数据源调用 ====================
    api = stock_api()

    # 1.1 连通性检查
    success, result = fetch_data_with_validation(
        api.ts_get_xxx_data,
        start_date, end_date,
        required_fields=['trade_date', 'open', 'close']
    )
    if not success:
        logging.error(f"数据拉取失败: {result}")
        return

    raw_data = result

    # ==================== 2. 数据清洗 ====================

    # 2.1 列名映射
    COLUMN_MAP = {
        'trade_date': 'trading_date',
        'ts_code': 'code',
        # ... 其他映射
    }
    data = map_column_names(raw_data, COLUMN_MAP)

    # 2.2 代码格式转换
    if 'code' in data.columns:
        data = data.with_columns(
            pl.col('code').map_elements(lambda x: convert_code_format(x, 'gm')).alias('code')
        )

    # 2.3 类型转换
    data = normalize_date_format(data, 'trading_date')
    data = cast_data_types(data, EXPECTED_SCHEMA)

    # 2.4 数据清洗
    data = clean_data(data)

    # ==================== 3. 数据对齐 ====================

    # 3.1 获取已有Schema
    existing_schema = get_parquet_dir_schema(save_dir)

    # 3.2 Schema对齐
    if existing_schema:
        data = align_schema(data, existing_schema)

    # ==================== 4. 增量合并 ====================

    if mode == 'insert':
        data, info = get_incremental_range(save_dir, data)
        logging.info(f"增量范围: {info}")
    elif mode == 'correct':
        correct_data(data, save_dir)
        return  # 修正模式直接返回，不走下面的写入逻辑

    # ==================== 5. 写入校验 ====================

    if data.height == 0:
        print("没有需要更新的数据")
        return

    expected_count = data.height
    if pre_write_validation(data):
        data.write_parquet(save_dir, partition_by=['trading_date'])
        post_write_validation(save_dir, expected_count)

    logging.info(f"XXX数据更新完成，共{data.height}条记录")
```

---

## 五、日志规范

### 5.1 日志文件
- 路径: `任务/log/数据更新.log`
- 编码: utf-8

### 5.2 日志级别
| 级别 | 使用场景 |
|------|----------|
| INFO | 正常流程进度、数据量统计 |
| WARNING | 数据异常但可继续、缺失字段 |
| ERROR | API失败、写入失败、校验失败 |
| CRITICAL | 认证失败、严重数据问题 |

### 5.3 日志模板
```python
logging.info(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] 开始更新日线数据...")
logging.info(f"数据源: tushare, 日期范围: {start_date} ~ {end_date}")
logging.info(f"拉取数据: {raw_data.height}条")
logging.info(f"清洗后: {cleaned_data.height}条")
logging.info(f"增量更新: {new_records.height}条")
logging.info(f"更新完成，耗时: {elapsed:.1f}秒")
```

---

## 六、注意事项

1. **Tushare限制**: 每分钟最多200次调用，注意限流
2. **掘金限制**: 需要连接miniQMT终端
3. **数据一致性**: 先更新日线，再基于日线更新其他数据
4. **复权因子**: 必须与日线数据同步更新，否则影响复权计算
5. **Schema变更**: API字段变化时，先适配再写入，切勿直接覆盖历史数据
6. **修正模式(correct)**: 会自动备份旧数据到 `xxx_backup_YYYYMMDD_HHMMSS` 目录
