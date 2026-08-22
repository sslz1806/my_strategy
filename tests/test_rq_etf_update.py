import datetime as dt
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pandas as pd
import polars as pl
import pytest

from my_utils import rq_fun


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_etf_update_module(module_name: str):
    path = PROJECT_ROOT / "任务" / "米筐ETF数据更新.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def sample_etf_instruments() -> pd.DataFrame:
    """构造同时包含存续、退市、未来上市和非法代码的完整 ETF 样本。"""
    return pd.DataFrame(
        {
            "order_book_id": [
                "510300.XSHG",
                "159901.XSHE",
                "510010.XSHG",
                "159077.XSHE",
                "BAD.CODE",
            ],
            "type": ["ETF", "ETF", "ETF", "ETF", "ETF"],
            "listed_date": [
                "2012-05-28",
                "2006-04-24",
                "2013-03-25",
                "2026-08-12",
                "2020-01-01",
            ],
            "de_listed_date": [
                "0000-00-00",
                "0000-00-00",
                "2020-12-31",
                "0000-00-00",
                "0000-00-00",
            ],
            "status": ["Active", "Active", "Delisted", "Unknown", "Active"],
        }
    )


def test_etf_universe_keeps_delisted_history_and_excludes_future_and_bad_codes():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())

    historical = rq_fun.filter_etf_codes_for_range(
        instruments, dt.date(2020, 1, 1), dt.date(2020, 12, 31)
    )
    current = rq_fun.filter_etf_codes_for_range(
        instruments, dt.date(2026, 8, 7), dt.date(2026, 8, 7)
    )

    assert historical == ["159901.XSHE", "510010.XSHG", "510300.XSHG"]
    assert current == ["159901.XSHE", "510300.XSHG"]
    assert instruments["listed_date"].map(type).eq(dt.date).all()
    assert instruments["de_listed_date"].map(type).eq(dt.date).all()


def test_etf_schemas_are_separate_from_stock_schemas():
    assert list(rq_fun.RQ_ETF_DAY_SCHEMA) == [
        "code",
        "trading_date",
        "pre_close",
        "open",
        "high",
        "low",
        "close",
        "change",
        "pct",
        "volume",
        "amount",
    ]
    assert list(rq_fun.RQ_ETF_MIN_SCHEMA) == [
        "code",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "trading_date",
    ]
    assert rq_fun.RQ_ETF_MIN_SCHEMA["datetime"] == pl.Datetime("us")


def test_normalize_etf_day_data_uses_reference_pre_close_and_rq_units():
    raw = pd.DataFrame(
        {
            "open": [11.0],
            "high": [11.25],
            "low": [10.88],
            "close": [11.24],
            "prev_close": [10.94],
            "volume": [203235546.0],
            "total_turnover": [2263042930.0],
        },
        index=pd.MultiIndex.from_arrays(
            [["510300.XSHG"], pd.to_datetime(["2026-06-12"])],
            names=["order_book_id", "date"],
        ),
    )

    result = rq_fun.normalize_etf_day_data(raw)

    assert result.schema == rq_fun.RQ_ETF_DAY_SCHEMA
    row = result.row(0, named=True)
    assert row["code"] == "SHSE.510300"
    assert row["pre_close"] == 10.94
    assert row["change"] == pytest.approx(0.30)
    assert row["pct"] == pytest.approx((11.24 / 10.94 - 1) * 100)
    assert row["volume"] == 203235546.0
    assert row["amount"] == 2263042930.0


def test_normalize_etf_minute_data_preserves_rq_end_timestamp_and_amount():
    raw = pd.DataFrame(
        {
            "open": [4.706, 4.713],
            "high": [4.716, 4.717],
            "low": [4.705, 4.711],
            "close": [4.713, 4.716],
            "volume": [20580149.0, 9736100.0],
            "total_turnover": [96958684.0, 45901975.0],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["510300.XSHG", "510300.XSHG"],
                pd.to_datetime(["2026-08-07 09:31:00", "2026-08-07 09:32:00"]),
            ],
            names=["order_book_id", "datetime"],
        ),
    )

    result = rq_fun.normalize_etf_minute_data(raw)

    assert result.schema == rq_fun.RQ_ETF_MIN_SCHEMA
    assert result["code"].to_list() == ["SHSE.510300", "SHSE.510300"]
    assert result["datetime"].to_list() == [
        dt.datetime(2026, 8, 7, 9, 31),
        dt.datetime(2026, 8, 7, 9, 32),
    ]
    assert result["amount"].to_list() == [96958684.0, 45901975.0]
    assert result["trading_date"].to_list() == [
        dt.date(2026, 8, 7),
        dt.date(2026, 8, 7),
    ]


def test_normalize_etf_data_rejects_missing_source_fields():
    raw = pd.DataFrame(
        {"close": [1.0]},
        index=pd.MultiIndex.from_arrays(
            [["510300.XSHG"], pd.to_datetime(["2026-08-07"])],
            names=["order_book_id", "date"],
        ),
    )

    with pytest.raises(ValueError, match="ETF day data missing columns"):
        rq_fun.normalize_etf_day_data(raw)


def test_build_etf_minute_batches_maximizes_days_within_row_budget():
    instruments = rq_fun.normalize_etf_instruments(
        pd.DataFrame(
            {
                "order_book_id": ["510300.XSHG", "159901.XSHE"],
                "type": ["ETF", "ETF"],
                "listed_date": ["2020-01-01", "2021-01-05"],
                "de_listed_date": ["0000-00-00", "0000-00-00"],
                "status": ["Active", "Active"],
            }
        )
    )
    days = [
        dt.date(2021, 1, 4),
        dt.date(2021, 1, 5),
        dt.date(2021, 1, 6),
        dt.date(2021, 1, 7),
    ]

    batches = rq_fun.build_etf_minute_batches(days, instruments, max_rows=720)

    assert [
        (batch.start_date, batch.end_date, batch.estimated_rows)
        for batch in batches
    ] == [
        (dt.date(2021, 1, 4), dt.date(2021, 1, 5), 720),
        (dt.date(2021, 1, 6), dt.date(2021, 1, 6), 480),
        (dt.date(2021, 1, 7), dt.date(2021, 1, 7), 480),
    ]
    assert batches[0].rq_codes == ("159901.XSHE", "510300.XSHG")


def test_build_etf_minute_batches_keeps_an_oversized_day_whole():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())
    day = dt.date(2020, 6, 1)

    batches = rq_fun.build_etf_minute_batches([day], instruments, max_rows=100)

    assert len(batches) == 1
    assert batches[0].trading_days == (day,)
    assert batches[0].estimated_rows == 3 * 240


def etf_day_frame(
    code: str = "SHSE.510300",
    trade_date: dt.date = dt.date(2026, 8, 7),
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "code": [code],
            "trading_date": [trade_date],
            "pre_close": [4.70],
            "open": [4.71],
            "high": [4.76],
            "low": [4.69],
            "close": [4.75],
            "change": [0.05],
            "pct": [0.05 / 4.70 * 100],
            "volume": [943535585.0],
            "amount": [4474693310.0],
        },
        schema=rq_fun.RQ_ETF_DAY_SCHEMA,
    )


def test_validate_etf_day_batch_rejects_unknown_code():
    day = dt.date(2026, 8, 7)

    with pytest.raises(RuntimeError, match="unexpected ETF codes"):
        rq_fun.validate_etf_day_batch(
            etf_day_frame(code="SHSE.999999"),
            ["510300.XSHG"],
            [day],
            today=dt.date(2026, 8, 9),
        )


def test_validate_etf_day_batch_rejects_duplicate_keys():
    day = dt.date(2026, 8, 7)

    with pytest.raises(RuntimeError, match="duplicate keys"):
        rq_fun.validate_etf_day_batch(
            pl.concat([etf_day_frame(), etf_day_frame()]),
            ["510300.XSHG"],
            [day],
            today=dt.date(2026, 8, 9),
        )


def test_validate_etf_day_batch_rejects_missing_historical_trading_day():
    day = dt.date(2026, 8, 7)

    with pytest.raises(RuntimeError, match="missing trading days"):
        rq_fun.validate_etf_day_batch(
            pl.DataFrame(schema=rq_fun.RQ_ETF_DAY_SCHEMA),
            ["510300.XSHG"],
            [day],
            today=dt.date(2026, 8, 9),
        )


def test_validate_etf_day_batch_rejects_date_outside_request():
    requested = dt.date(2026, 8, 7)
    outside = dt.date(2026, 8, 8)

    with pytest.raises(RuntimeError, match="outside requested dates"):
        rq_fun.validate_etf_day_batch(
            etf_day_frame(trade_date=outside),
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )


def etf_minute_frame(trade_date: dt.date, count: int) -> pl.DataFrame:
    session_times = [
        dt.datetime.combine(trade_date, dt.time(9, 31)) + dt.timedelta(minutes=index)
        for index in range(120)
    ] + [
        dt.datetime.combine(trade_date, dt.time(13, 1)) + dt.timedelta(minutes=index)
        for index in range(120)
    ]
    datetimes = session_times[:count]
    return pl.DataFrame(
        {
            "code": ["SHSE.510300"] * count,
            "datetime": datetimes,
            "open": [1.0] * count,
            "high": [1.0] * count,
            "low": [1.0] * count,
            "close": [1.0] * count,
            "volume": [0.0] * count,
            "amount": [0.0] * count,
            "trading_date": [trade_date] * count,
        },
        schema=rq_fun.RQ_ETF_MIN_SCHEMA,
    )


def raw_etf_minute_data(
    codes: list[str], trading_days: list[dt.date]
) -> pd.DataFrame:
    """构造与 rqdatac.get_price 一致的完整分钟 MultiIndex 响应。"""
    timestamps = []
    for trade_date in trading_days:
        timestamps.extend(
            dt.datetime.combine(trade_date, dt.time(9, 31))
            + dt.timedelta(minutes=index)
            for index in range(120)
        )
        timestamps.extend(
            dt.datetime.combine(trade_date, dt.time(13, 1))
            + dt.timedelta(minutes=index)
            for index in range(120)
        )
    row_count = len(codes) * len(timestamps)
    return pd.DataFrame(
        {
            "open": [1.0] * row_count,
            "high": [1.0] * row_count,
            "low": [1.0] * row_count,
            "close": [1.0] * row_count,
            "volume": [10.0] * row_count,
            "total_turnover": [10.0] * row_count,
        },
        index=pd.MultiIndex.from_product(
            [codes, pd.to_datetime(timestamps)],
            names=["order_book_id", "datetime"],
        ),
    )


def test_drop_incomplete_current_minute_date_keeps_completed_history():
    historical = dt.date(2026, 8, 7)
    today = dt.date(2026, 8, 10)
    data = pl.concat(
        [etf_minute_frame(historical, 240), etf_minute_frame(today, 30)]
    )

    result = rq_fun.drop_incomplete_current_etf_minute_date(data, today=today)

    assert result.height == 240
    assert result["trading_date"].unique().to_list() == [historical]


def test_drop_incomplete_current_minute_date_keeps_completed_today():
    today = dt.date(2026, 8, 10)
    data = etf_minute_frame(today, 240)

    result = rq_fun.drop_incomplete_current_etf_minute_date(data, today=today)

    assert result.equals(data)


def test_validate_etf_minute_batch_rejects_date_outside_request():
    requested = dt.date(2026, 8, 7)
    outside = dt.date(2026, 8, 8)

    with pytest.raises(RuntimeError, match="outside requested dates"):
        rq_fun.validate_etf_minute_batch(
            etf_minute_frame(outside, 1),
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )


def test_validate_etf_minute_batch_rejects_datetime_partition_mismatch():
    requested = dt.date(2026, 8, 7)
    data = etf_minute_frame(dt.date(2026, 8, 8), 1).with_columns(
        pl.lit(requested).cast(pl.Date).alias("trading_date")
    )

    with pytest.raises(RuntimeError, match="does not match trading_date"):
        rq_fun.validate_etf_minute_batch(
            data,
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )


def test_validate_etf_minute_batch_rejects_missing_historical_trading_day():
    requested = dt.date(2026, 8, 7)

    with pytest.raises(RuntimeError, match="missing trading days"):
        rq_fun.validate_etf_minute_batch(
            pl.DataFrame(schema=rq_fun.RQ_ETF_MIN_SCHEMA),
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )


def test_validate_etf_minute_batch_rejects_duplicate_keys():
    requested = dt.date(2026, 8, 7)
    duplicated = pl.concat(
        [etf_minute_frame(requested, 1), etf_minute_frame(requested, 1)]
    )

    with pytest.raises(RuntimeError, match="duplicate keys"):
        rq_fun.validate_etf_minute_batch(
            duplicated,
            ["510300.XSHG"],
            [requested],
            today=dt.date(2026, 8, 9),
        )


def test_rqdata_etf_adapter_uses_one_unadjusted_call_for_all_codes():
    from my_utils import rqdata

    day_result = pd.DataFrame({"close": [1.0]})
    minute_result = pd.DataFrame({"close": [1.0]})
    with patch.object(rqdata.rq, "init"), patch.object(
        rqdata.rq, "get_price", side_effect=[day_result, minute_result]
    ) as get_price:
        source = rqdata.RqData()
        codes = ["510300.XSHG", "159915.XSHE"]
        day = source.fetch_etf_day_range(
            codes, dt.date(2026, 8, 7), dt.date(2026, 8, 7)
        )
        minute = source.fetch_etf_minute_range(
            codes, dt.date(2026, 8, 7), dt.date(2026, 8, 7)
        )

    assert day is day_result
    assert minute is minute_result
    assert get_price.call_args_list == [
        call(
            codes,
            start_date=dt.date(2026, 8, 7),
            end_date=dt.date(2026, 8, 7),
            frequency="1d",
            fields=rqdata.ETF_DAY_FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        ),
        call(
            codes,
            start_date=dt.date(2026, 8, 7),
            end_date=dt.date(2026, 8, 7),
            frequency="1m",
            fields=rqdata.ETF_MINUTE_FIELDS,
            adjust_type="none",
            skip_suspended=False,
            expect_df=True,
        ),
    ]


def test_rqdata_etf_metadata_and_calendar_are_single_calls():
    from my_utils import rqdata

    instruments = sample_etf_instruments()
    with patch.object(rqdata.rq, "init"), patch.object(
        rqdata.rq, "all_instruments", return_value=instruments
    ) as all_instruments, patch.object(
        rqdata.rq,
        "get_trading_dates",
        return_value=[pd.Timestamp("2026-08-06"), pd.Timestamp("2026-08-07")],
    ) as get_trading_dates:
        source = rqdata.RqData()
        result_instruments = source.get_etf_instruments()
        days = source.get_trading_days(
            dt.date(2026, 8, 6), dt.date(2026, 8, 7)
        )

    assert result_instruments is instruments
    assert days == [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]
    all_instruments.assert_called_once_with(type="ETF", market="cn")
    get_trading_dates.assert_called_once_with(
        dt.date(2026, 8, 6), dt.date(2026, 8, 7), market="cn"
    )


def test_fetch_and_save_bisects_gateway_error_by_trading_day():
    module = load_etf_update_module("rq_etf_fallback_split_test")
    days = [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]
    fetch_calls = []
    consumed = []

    def fetch_once(batch_days):
        fetch_calls.append(tuple(batch_days))
        if len(batch_days) == 2:
            raise module.GatewayError("response too large")
        return pd.DataFrame({"day": batch_days})

    def consume(raw, batch_days):
        consumed.append(tuple(batch_days))
        return len(raw)

    written = module.fetch_and_save(fetch_once, consume, days)

    assert written == 2
    assert fetch_calls == [(days[0], days[1]), (days[0],), (days[1],)]
    assert consumed == [(days[0],), (days[1],)]


def test_fetch_and_save_retries_network_once():
    module = load_etf_update_module("rq_etf_fallback_network_retry_test")
    day = dt.date(2026, 8, 7)
    fetch_once = MagicMock(
        side_effect=[ConnectionError("connection reset"), pd.DataFrame({"x": [1]})]
    )
    sleeps = []

    written = module.fetch_and_save(
        fetch_once,
        lambda raw, _days: len(raw),
        [day],
        sleep_func=sleeps.append,
    )

    assert written == 1
    assert fetch_once.call_count == 2
    assert sleeps == [3.0]


@pytest.mark.parametrize(
    "error_type",
    [
        "AuthenticationFailed",
        "PermissionDenied",
        "QuotaExceeded",
    ],
)
def test_fetch_and_save_does_not_retry_fatal_errors(error_type):
    module = load_etf_update_module(f"rq_etf_fallback_fatal_{error_type}")
    error_class = getattr(module, error_type)
    fetch_once = MagicMock(side_effect=error_class("fatal"))

    with pytest.raises(error_class):
        module.fetch_and_save(
            fetch_once,
            MagicMock(),
            [dt.date(2026, 8, 7)],
            sleep_func=MagicMock(),
        )

    fetch_once.assert_called_once()


def test_fetch_and_save_aborts_when_single_day_is_still_too_large():
    module = load_etf_update_module("rq_etf_fallback_single_day_test")
    fetch_once = MagicMock(side_effect=module.GatewayError("response too large"))

    with pytest.raises(module.GatewayError):
        module.fetch_and_save(
            fetch_once,
            MagicMock(),
            [dt.date(2026, 8, 7)],
        )

    fetch_once.assert_called_once()


def test_fetch_and_save_does_not_bisect_validation_or_save_errors():
    module = load_etf_update_module("rq_etf_fallback_consume_error_test")
    days = [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]
    fetch_once = MagicMock(return_value=pd.DataFrame({"x": [1]}))

    with pytest.raises(RuntimeError, match="validation failed"):
        module.fetch_and_save(
            fetch_once,
            MagicMock(side_effect=RuntimeError("validation failed")),
            days,
        )

    fetch_once.assert_called_once_with(days)


def test_update_day_requests_all_active_codes_once_and_writes(tmp_path):
    module = load_etf_update_module("rq_etf_day_pipeline_test")
    trade_date = dt.date(2026, 8, 7)
    codes = ["159901.XSHE", "510300.XSHG"]
    raw = pd.DataFrame(
        {
            "open": [1.0, 4.7],
            "high": [1.1, 4.8],
            "low": [0.9, 4.6],
            "close": [1.05, 4.75],
            "prev_close": [1.0, 4.7],
            "volume": [100.0, 200.0],
            "total_turnover": [105.0, 950.0],
        },
        index=pd.MultiIndex.from_product(
            [codes, pd.to_datetime([trade_date])],
            names=["order_book_id", "date"],
        ),
    )
    source = MagicMock()
    source.fetch_etf_day_range.return_value = raw

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_day(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            [trade_date],
            mode="update",
        )

    assert written == 2
    source.fetch_etf_day_range.assert_called_once_with(codes, trade_date, trade_date)
    output = pl.read_parquet(
        str(
            tmp_path
            / module.RQ_ETF_DAY_DIR
            / f"trading_date={trade_date.isoformat()}"
            / "*.parquet"
        )
    )
    assert output.schema == rq_fun.RQ_ETF_DAY_SCHEMA
    assert output["code"].sort().to_list() == ["SHSE.510300", "SZSE.159901"]


def test_update_minute_keeps_original_times_and_all_codes(tmp_path):
    module = load_etf_update_module("rq_etf_minute_pipeline_test")
    trade_date = dt.date(2026, 8, 7)
    codes = ["159901.XSHE", "510300.XSHG"]
    session_times = [
        dt.datetime.combine(trade_date, dt.time(9, 31)) + dt.timedelta(minutes=index)
        for index in range(120)
    ] + [
        dt.datetime.combine(trade_date, dt.time(13, 1)) + dt.timedelta(minutes=index)
        for index in range(120)
    ]
    row_count = len(codes) * len(session_times)
    raw = pd.DataFrame(
        {
            "open": [1.0] * row_count,
            "high": [1.0] * row_count,
            "low": [1.0] * row_count,
            "close": [1.0] * row_count,
            "volume": [10.0] * row_count,
            "total_turnover": [10.0] * row_count,
        },
        index=pd.MultiIndex.from_product(
            [codes, pd.to_datetime(session_times)],
            names=["order_book_id", "datetime"],
        ),
    )
    source = MagicMock()
    source.get_quota.return_value = {
        "bytes_limit": 0,
        "bytes_used": 0,
        "remaining_days": 0,
        "license_type": "FULL",
    }
    source.fetch_etf_minute_range.return_value = raw

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            [trade_date],
            mode="update",
            max_rows=3_000_000,
        )

    assert written == 480
    source.fetch_etf_minute_range.assert_called_once_with(codes, trade_date, trade_date)
    output = pl.read_parquet(
        str(
            tmp_path
            / module.RQ_ETF_MIN_DIR
            / f"trading_date={trade_date.isoformat()}"
            / "*.parquet"
        )
    )
    assert output.schema == rq_fun.RQ_ETF_MIN_SCHEMA
    assert output["datetime"].min().time() == dt.time(9, 31)
    assert output["datetime"].max().time() == dt.time(15, 0)


def test_etf_update_parse_args_defaults():
    module = load_etf_update_module("rq_etf_parse_defaults_test")

    args = module.parse_args([])

    assert args.start_date == "2018-01-01"
    assert args.end_date == dt.date.today().isoformat()
    assert args.mode == "insert"
    assert args.data_type == "all"
    assert args.max_minute_rows == 3_000_000
    assert args.quota_reserve_mb == 128


@pytest.mark.parametrize("data_type", ["day", "min", "all"])
def test_etf_update_parse_args_accepts_each_data_type(data_type):
    module = load_etf_update_module(f"rq_etf_parse_{data_type}_test")

    args = module.parse_args(
        [
            "--data-type",
            data_type,
            "--mode",
            "update",
            "--max-minute-rows",
            "480000",
        ]
    )

    assert args.data_type == data_type
    assert args.mode == "update"
    assert args.max_minute_rows == 480_000


def test_etf_update_parse_args_rejects_non_positive_minute_budget():
    module = load_etf_update_module("rq_etf_parse_bad_budget_test")

    with pytest.raises(SystemExit):
        module.parse_args(["--max-minute-rows", "0"])


def test_etf_update_main_uses_independent_day_and_minute_cursors():
    module = load_etf_update_module("rq_etf_main_independent_cursor_test")
    requested_start = dt.date(2021, 1, 1)
    minute_start = dt.date(2026, 8, 6)
    end_date = dt.date(2026, 8, 7)
    trading_days = [minute_start, end_date]
    source = MagicMock()
    source.get_etf_instruments.return_value = sample_etf_instruments()
    source.get_trading_days.return_value = trading_days

    with patch.object(module, "get_logger"), patch.object(
        module, "infer_start_date", side_effect=[None, minute_start]
    ) as infer_start, patch.object(module, "RqData", return_value=source), patch.object(
        module, "update_day"
    ) as update_day, patch.object(
        module, "update_minute", return_value=480
    ) as update_minute:
        result = module.main(
            [
                "--start-date",
                requested_start.isoformat(),
                "--end-date",
                end_date.isoformat(),
                "--data-type",
                "all",
                "--max-minute-rows",
                "500000",
            ]
        )

    assert result == 0
    assert infer_start.call_args_list == [
        call(requested_start, module.RQ_ETF_DAY_DIR, "insert", end_date=end_date),
        call(requested_start, module.RQ_ETF_MIN_DIR, "insert", end_date=end_date),
    ]
    source.get_etf_instruments.assert_called_once_with()
    source.get_trading_days.assert_called_once_with(minute_start, end_date)
    update_day.assert_not_called()
    update_minute.assert_called_once()
    minute_call = update_minute.call_args
    assert minute_call.args[0] is source
    assert minute_call.args[2] == trading_days
    assert minute_call.kwargs == {
        "mode": "insert",
        "max_rows": 500_000,
        "quota_reserve_bytes": 128 * 1024 * 1024,
    }


def test_etf_update_main_returns_before_official_calls_when_all_selected_are_current():
    module = load_etf_update_module("rq_etf_main_noop_test")

    with patch.object(module, "get_logger"), patch.object(
        module, "infer_start_date", side_effect=[None, None]
    ), patch.object(module, "RqData") as source_class:
        result = module.main(
            [
                "--start-date",
                "2021-01-01",
                "--end-date",
                "2026-08-07",
                "--data-type",
                "all",
            ]
        )

    assert result == 0
    source_class.assert_not_called()


def test_rqdata_get_quota_uses_official_account_api_once():
    from my_utils import rqdata

    expected = {
        "bytes_limit": 1_073_741_824,
        "bytes_used": 123_456,
        "remaining_days": 14,
        "license_type": "TRIAL",
    }
    with patch.object(rqdata.rq, "init"), patch.object(
        rqdata.rq.user, "get_quota", return_value=expected
    ) as get_quota:
        result = rqdata.RqData().get_quota()

    assert result == expected
    get_quota.assert_called_once_with()


@pytest.mark.parametrize(
    ("quota", "reserve_bytes", "expected"),
    [
        ({"bytes_limit": 1_000, "bytes_used": 250}, 100, 650),
        ({"bytes_limit": 1_000, "bytes_used": 950}, 100, 0),
        ({"bytes_limit": 0, "bytes_used": 999}, 100, None),
    ],
)
def test_quota_remaining_bytes_reserves_capacity(quota, reserve_bytes, expected):
    assert rq_fun.quota_remaining_bytes(quota, reserve_bytes) == expected


def test_measure_bytes_per_row_uses_official_counter_delta():
    result = rq_fun.measure_bytes_per_row(
        {"bytes_used": 100},
        {"bytes_used": 580},
        row_count=240,
    )

    assert result == 2.0


@pytest.mark.parametrize(
    ("before", "after", "row_count", "message"),
    [
        ({"bytes_used": 100}, {"bytes_used": 200}, 0, "row_count"),
        ({"bytes_used": 100}, {"bytes_used": 100}, 240, "did not increase"),
        ({"bytes_used": 100}, {"bytes_used": 90}, 240, "did not increase"),
    ],
)
def test_measure_bytes_per_row_rejects_invalid_calibration(
    before, after, row_count, message
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        rq_fun.measure_bytes_per_row(before, after, row_count)


def test_select_minute_days_for_quota_returns_longest_continuous_prefix():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())
    days = [dt.date(2026, 8, 6), dt.date(2026, 8, 7)]

    selected = rq_fun.select_minute_days_for_quota(
        days,
        instruments,
        available_bytes=1_200,
        bytes_per_row=2.0,
        safety_factor=1.25,
    )

    # 每天 2 只 ETF × 240 行 × 2 字节 × 1.25 安全系数 = 1,200 字节。
    assert selected == [days[0]]


def test_select_minute_days_for_unlimited_quota_returns_all_sorted_days():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())
    earlier = dt.date(2026, 8, 6)
    later = dt.date(2026, 8, 7)

    selected = rq_fun.select_minute_days_for_quota(
        [later, earlier, later],
        instruments,
        available_bytes=None,
        bytes_per_row=2.0,
    )

    assert selected == [earlier, later]


def test_select_minute_days_for_quota_returns_empty_when_first_day_does_not_fit():
    instruments = rq_fun.normalize_etf_instruments(sample_etf_instruments())

    selected = rq_fun.select_minute_days_for_quota(
        [dt.date(2026, 8, 7)],
        instruments,
        available_bytes=1_199,
        bytes_per_row=2.0,
        safety_factor=1.25,
    )

    assert selected == []


def test_cleanup_new_failed_partitions_preserves_preexisting_date(tmp_path):
    old_day = dt.date(2026, 8, 6)
    new_day = dt.date(2026, 8, 7)
    save_dir = "minute"
    old_partition = tmp_path / save_dir / f"trading_date={old_day}"
    new_partition = tmp_path / save_dir / f"trading_date={new_day}"
    staging_dir = tmp_path / save_dir / ".rq-staging-interrupted"
    old_partition.mkdir(parents=True)
    new_partition.mkdir(parents=True)
    staging_dir.mkdir(parents=True)

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        removed = rq_fun.cleanup_new_failed_partitions(
            save_dir,
            [old_day, new_day],
            dates_before_run={old_day},
        )

    assert removed == [new_day]
    assert old_partition.exists()
    assert not new_partition.exists()
    assert not staging_dir.exists()


def test_cleanup_new_failed_partitions_is_noop_for_missing_partition(tmp_path):
    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        removed = rq_fun.cleanup_new_failed_partitions(
            "minute",
            [dt.date(2026, 8, 7)],
            dates_before_run=set(),
        )

    assert removed == []


def test_cleanup_new_failed_partitions_rejects_path_outside_data_root(tmp_path):
    data_root = tmp_path / "root"
    outside = tmp_path / "outside"
    outside_partition = outside / "trading_date=2026-08-07"
    outside_partition.mkdir(parents=True)

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(data_root)), pytest.raises(
        ValueError, match="outside DATA_ROOT_DIR"
    ):
        rq_fun.cleanup_new_failed_partitions(
            "../outside",
            [dt.date(2026, 8, 7)],
            dates_before_run=set(),
        )

    assert outside_partition.exists()


def test_update_minute_calibrates_and_writes_only_quota_fitting_prefix(tmp_path):
    module = load_etf_update_module("rq_etf_quota_calibration_test")
    days = [
        dt.date(2026, 8, 5),
        dt.date(2026, 8, 6),
        dt.date(2026, 8, 7),
    ]
    codes = ["159901.XSHE", "510300.XSHG"]
    source = MagicMock()
    source.get_quota.side_effect = [
        {
            "bytes_limit": 3_080,
            "bytes_used": 1_000,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
        {
            "bytes_limit": 3_080,
            "bytes_used": 1_480,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
    ]
    source.fetch_etf_minute_range.side_effect = (
        lambda requested_codes, start, end: raw_etf_minute_data(
            requested_codes,
            [day for day in days if start <= day <= end],
        )
    )

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            days,
            mode="insert",
            max_rows=3_000_000,
            quota_reserve_bytes=1_000,
            quota_safety_factor=1.25,
            calibration_bytes_per_row=1.0,
        )

    assert written == 960
    partitions = sorted(
        path.name
        for path in (tmp_path / module.RQ_ETF_MIN_DIR).glob("trading_date=*")
    )
    assert partitions == [
        "trading_date=2026-08-05",
        "trading_date=2026-08-06",
    ]
    assert source.fetch_etf_minute_range.call_count == 2
    assert source.get_quota.call_count == 2


def test_update_minute_stops_before_price_call_when_safe_quota_is_empty(tmp_path):
    module = load_etf_update_module("rq_etf_quota_empty_test")
    source = MagicMock()
    source.get_quota.return_value = {
        "bytes_limit": 1_000,
        "bytes_used": 900,
        "remaining_days": 14,
        "license_type": "TRIAL",
    }

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            [dt.date(2026, 8, 7)],
            mode="insert",
            quota_reserve_bytes=100,
        )

    assert written == 0
    source.fetch_etf_minute_range.assert_not_called()


def test_update_minute_quota_error_removes_only_new_failed_partition(tmp_path):
    module = load_etf_update_module("rq_etf_quota_cleanup_test")
    trade_date = dt.date(2026, 8, 7)
    failed_partition = (
        tmp_path / module.RQ_ETF_MIN_DIR / f"trading_date={trade_date}"
    )
    source = MagicMock()
    source.get_quota.return_value = {
        "bytes_limit": 10_000,
        "bytes_used": 100,
        "remaining_days": 14,
        "license_type": "TRIAL",
    }

    def fail_after_creating_bad_partition(*_args):
        failed_partition.mkdir(parents=True)
        raise module.QuotaExceeded("traffic exhausted")

    source.fetch_etf_minute_range.side_effect = fail_after_creating_bad_partition

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)), pytest.raises(
        module.QuotaExceeded
    ):
        module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            [trade_date],
            mode="insert",
            quota_reserve_bytes=100,
            calibration_bytes_per_row=1.0,
        )

    assert not failed_partition.exists()


def test_update_minute_waits_for_delayed_quota_counter(tmp_path):
    module = load_etf_update_module("rq_etf_delayed_quota_test")
    trade_date = dt.date(2026, 8, 7)
    codes = ["159901.XSHE", "510300.XSHG"]
    source = MagicMock()
    source.get_quota.side_effect = [
        {
            "bytes_limit": 10_000,
            "bytes_used": 1_000,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
        {
            "bytes_limit": 10_000,
            "bytes_used": 1_000,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
        {
            "bytes_limit": 10_000,
            "bytes_used": 1_480,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
    ]
    source.fetch_etf_minute_range.return_value = raw_etf_minute_data(
        codes,
        [trade_date],
    )
    sleeps = []

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            [trade_date],
            mode="insert",
            quota_reserve_bytes=100,
            calibration_bytes_per_row=1.0,
            quota_poll_attempts=3,
            quota_poll_seconds=2.0,
            quota_sleep_func=sleeps.append,
        )

    assert written == 480
    assert sleeps == [2.0]
    assert source.get_quota.call_count == 3
    assert source.fetch_etf_minute_range.call_count == 1


def test_update_minute_uses_largest_safe_row_batch_for_calibration(tmp_path):
    module = load_etf_update_module("rq_etf_multi_day_calibration_test")
    days = [
        dt.date(2026, 8, 5),
        dt.date(2026, 8, 6),
        dt.date(2026, 8, 7),
    ]
    codes = ["159901.XSHE", "510300.XSHG"]
    source = MagicMock()
    source.get_quota.side_effect = [
        {
            "bytes_limit": 100_000,
            "bytes_used": 1_000,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
        {
            "bytes_limit": 100_000,
            "bytes_used": 1_960,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
    ]
    source.fetch_etf_minute_range.side_effect = (
        lambda requested_codes, start, end: raw_etf_minute_data(
            requested_codes,
            [day for day in days if start <= day <= end],
        )
    )

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            days,
            mode="insert",
            max_rows=960,
            quota_reserve_bytes=1_000,
            calibration_bytes_per_row=1.0,
        )

    assert written == 1_440
    first_call = source.fetch_etf_minute_range.call_args_list[0]
    assert first_call.args == (codes, days[0], days[1])
    assert source.fetch_etf_minute_range.call_count == 2


def test_update_minute_default_margin_tolerates_later_row_cost_increase(tmp_path):
    module = load_etf_update_module("rq_etf_default_quota_margin_test")
    days = [
        dt.date(2026, 8, 5),
        dt.date(2026, 8, 6),
        dt.date(2026, 8, 7),
    ]
    codes = ["159901.XSHE", "510300.XSHG"]
    source = MagicMock()
    source.get_quota.side_effect = [
        {
            "bytes_limit": 3_780,
            "bytes_used": 1_000,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
        {
            "bytes_limit": 3_780,
            "bytes_used": 1_480,
            "remaining_days": 14,
            "license_type": "TRIAL",
        },
    ]
    source.fetch_etf_minute_range.side_effect = (
        lambda requested_codes, start, end: raw_etf_minute_data(
            requested_codes,
            [day for day in days if start <= day <= end],
        )
    )

    with patch.object(rq_fun, "DATA_ROOT_DIR", str(tmp_path)):
        written = module.update_minute(
            source,
            rq_fun.normalize_etf_instruments(sample_etf_instruments()),
            days,
            mode="insert",
            max_rows=480,
            quota_reserve_bytes=1_000,
            calibration_bytes_per_row=1.0,
        )

    # 校准值是 1 字节/行；默认 1.75 倍余量只允许再取一天，即使后续
    # 实际上涨 40% 仍不会突破计划额度。1.25 倍会错误地把两天都纳入。
    assert written == 960
    partitions = sorted(
        path.name
        for path in (tmp_path / module.RQ_ETF_MIN_DIR).glob("trading_date=*")
    )
    assert partitions == [
        "trading_date=2026-08-05",
        "trading_date=2026-08-06",
    ]
