import sys
import time
import pandas as pd
from xtquant.xttrader import XtQuantTrader #创建交易对象使用
from xtquant.xttype import StockAccount #订阅账户信息使用
from xtquant import xtconstant #执行交易的时候需要引入
from datetime import datetime #时间戳改为日期时间格式的时候使用
# 回调类,处理账户状态
from xtquant.xttrader import XtQuantTraderCallback
from my_utils.fun import get_logger
from my_utils.qmt_process import (
    connect_trader_with_retry,
    get_process_name_from_path,
    is_process_running as qmt_is_process_running,
    start_software,
    stop_software,
)
logging = get_logger(log_file='log/实盘.log',inherit=False)

class MyXtQuantTraderCallback(XtQuantTraderCallback):
    def on_disconnected(self):
        """
        连接断开
        :return:
        """
        logging.info(datetime.datetime.now(),'连接断开回调')

    def on_stock_order(self, order):
        """
        委托回报推送
        :param order: XtOrder对象
        :return:
        """
        logging.info(datetime.datetime.now(), '委托回调', order.order_remark)


    def on_stock_trade(self, trade):
        """
        成交变动推送
        :param trade: XtTrade对象
        :return:
        """
        logging.info(datetime.datetime.now(), '成交回调', trade.order_remark)


    def on_order_error(self, order_error):
        """
        委托失败推送
        :param order_error:XtOrderError 对象
        :return:
        """
        # logging.info("on order_error callback")
        # logging.info(order_error.order_id, order_error.error_id, order_error.error_msg)
        logging.info(f"委托报错回调 {order_error.order_remark} {order_error.error_msg}")

    def on_cancel_error(self, cancel_error):
        """
        撤单失败推送
        :param cancel_error: XtCancelError 对象
        :return:
        """
        logging.info(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_order_stock_async_response(self, response):
        """
        异步下单回报推送
        :param response: XtOrderResponse 对象
        :return:
        """
        logging.info(f"异步委托回调 {response.order_remark}")

    def on_cancel_order_stock_async_response(self, response):
        """
        收到撤单回调信息
        :param response: XtCancelOrderResponse 对象
        :return:
        """
        logging.info(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_account_status(self, status):
        """
        账号状态信息变动推送
        :param response: XtAccountStatus 对象
        :return:
        """
        logging.info(datetime.datetime.now(), sys._getframe().f_code.co_name)

    def on_stock_position(self, position):
        """
        持仓变动推送，根据：https://blog.csdn.net/liuyukuan/article/details/128754695
        :param position: XtPosition对象
        :return:
        """
        logging.info("on position callback")
        logging.info(position.stock_code, position.volume)

    def on_connected(self):
            """
            连接成功推送
            """
            pass

    def on_stock_asset(self,asset):
            """
            资金变动推送，根据：https://blog.csdn.net/liuyukuan/article/details/128754695
            :param asset: XtAsset对象
            :return:
            """
            logging.info("资金变动推送on asset callback")
            logging.info(asset.account_id,asset.cash,asset.total_asset)

# 委托信息
def orders_df():
    orders_df = pd.DataFrame([(order.stock_code, order.order_volume, order.price, order.order_id, order.status_msg,
                                datetime.fromtimestamp(order.order_time).strftime('%H:%M:%S'))
                                for order in xt_trader.query_stock_orders(ID)],
                                columns=['证券代码', '委托数量', '委托价格', '订单编号','委托状态','报单时间'])
    return orders_df

# 成交信息
def trades_df():
    trades_df = pd.DataFrame([(trade.stock_code, trade.traded_volume, trade.traded_price,trade.traded_amount,trade.order_id, trade.traded_id,
                                datetime.fromtimestamp(trade.traded_time).strftime('%H:%M:%S'))
                                for trade in xt_trader.query_stock_trades(ID)],
                                columns=['证券代码', '成交数量', '成交均价','成交金额','订单编号', '成交编号', '成交时间'])
    return trades_df
def positions_df():
    positions_df = pd.DataFrame([(position.stock_code, position.volume, position.can_use_volume, position.frozen_volume,
                                    position.open_price, position.market_value, position.on_road_volume, position.yesterday_volume)
                                    for position in xt_trader.query_stock_positions(ID)],
                                columns=['证券代码', '持仓数量', '可用数量', '冻结数量', '开仓价格', '持仓市值', '在途股份', '昨夜持股'])
    return positions_df

#——————————————————————————————————————————————————————————————————————————————————————————————————————
#设置你的path='' 文件夹userdata_mini前面改为自己的QMT安装路径信息，acc=''引号内填入自己的账号
path = r'F:\trading\东北证券NET专业版\userdata_mini'
acct = "51318497"

# QMT 客户端可执行文件路径（用于自动检测/启动客户端）
QMT_CLIENT_EXE = r'F:\trading\东北证券NET专业版\bin.x64\XtItClient.exe'
# XtItClient.exe 只是登录/启动器，登录后会退出并拉起长期运行的 XtMiniQmt.exe。
# 健康检测必须跟踪后者，否则会把已运行的 QMT 误判为未启动并重复拉起。
QMT_PROCESS_NAME = 'XtMiniQmt.exe'


def start_qmt_client(exe_path, process_name, max_wait=60, post_start_delay=2):
    """
    如果 QMT 客户端进程未运行，则启动它并等待初始化完成。

    注意：本函数只能拉起客户端程序，无法自动完成账号登录/验证。
    若 QMT 启动后需要手动登录，请在运行脚本前先登录，或等待登录完成后再继续。
    """
    process_name = process_name or get_process_name_from_path(exe_path)
    if qmt_is_process_running(process_name):
        logging.info(f'已检测到 QMT 客户端进程 [{process_name}]，无需启动')
        return True

    logging.info(f'未检测到 QMT 客户端，正在启动: {exe_path}')
    if not start_software(
        exe_path,
        avoid_duplicate=True,
        show_window=True,
        process_name=process_name,
        max_wait=max_wait,
        poll_interval=2,
    ):
        logging.error(f'启动 QMT 客户端失败: {exe_path}')
        return False

    logging.info(f'QMT 主进程 [{process_name}] 已启动')

    # 主进程出现后给交易服务少量初始化时间；真正的健康状态由 connect() 验证。
    if post_start_delay > 0:
        logging.info(f'QMT 客户端刚启动，等待 {post_start_delay} 秒完成初始化...')
        time.sleep(post_start_delay)

    return True


def stop_qmt_client(target=QMT_CLIENT_EXE, force=False, timeout=5):
    """
    关闭 QMT 客户端进程

    :param target: 进程名/EXE路径/快捷方式路径，默认使用 QMT_CLIENT_EXE
    :param force: 是否强制终止
    :param timeout: 等待进程终止的最大秒数
    :return: 关闭成功的进程数
    """
    killed_count = stop_software(target, force=force, timeout=timeout)
    if killed_count > 0:
        logging.info(f'已关闭 QMT 客户端进程，共 {killed_count} 个')
    else:
        logging.info('未检测到运行中的 QMT 客户端进程')
    return killed_count


# 每次重试使用不同会话号。保留 session_id/callback 全局变量，兼容既有
# ``from my_utils.my_qmt import *`` 调用方。
_next_session_id = int(time.time())
session_id = None
callback = None


def _create_xt_trader():
    """在 QMT 主进程就绪后创建一个新的 xtquant 会话。"""
    global _next_session_id, session_id, callback
    session_id = _next_session_id
    _next_session_id += 1
    trader = XtQuantTrader(path, session_id)
    callback = MyXtQuantTraderCallback()
    trader.register_callback(callback)
    return trader


def _ensure_qmt_client():
    """连接前及每次重试前重新检测 QMT，必要时自动拉起。"""
    return start_qmt_client(QMT_CLIENT_EXE, QMT_PROCESS_NAME)


# start() 只启动 xtquant 的异步线程；connect() 才执行真正的终端连接。
# 失败时会清理当前 xtquant 会话、重新检测 QMT，并创建新会话重试。
xt_trader = connect_trader_with_retry(
    _create_xt_trader,
    _ensure_qmt_client,
    max_attempts=7,
    retry_interval=4,
    logger=logging,
)
logging.info('【软件终端连接成功！】')


#——————————————————————————————————————————————————————————————————————————————————————————————————————
# 4.订阅账户信息
ID = StockAccount(acct)
subscribe_result = xt_trader.subscribe(ID)
if subscribe_result == 0:logging.info('【账户信息订阅成功！】')
else:
    logging.info('【账户信息订阅失败！】','\n 账户配置错误，检查账号是否正确。','\n acct=""内填加你的账号')
    sys.exit() #如果运行环境，账户都没配置好，后面的代码就不执行


#打印账户信息
if __name__ == "__main__":
    asset = xt_trader.query_stock_asset(ID)
    print('-'*18,'【{0}】'.format(asset.account_id),'-'*18)
    if asset:print(f"资产总额: {asset.total_asset}\n"
                    f"持仓市值：{asset.market_value}\n"
                    f"可用资金：{asset.cash}\n"
                    f"在途资金：{asset.frozen_cash}")
