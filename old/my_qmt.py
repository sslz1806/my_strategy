import sys
import time
import pandas as pd
from xtquant.xttrader import XtQuantTrader #创建交易对象使用
from xtquant.xttype import StockAccount #订阅账户信息使用
from xtquant import xtconstant #执行交易的时候需要引入
from datetime import datetime #时间戳改为日期时间格式的时候使用
# 回调类,处理账户状态
from xtquant.xttrader import XtQuantTraderCallback
from fun import get_logger
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
# 1.创建交易对象API实例
session_id = int(time.time())
xt_trader = XtQuantTrader(path, session_id)
# 2.创建并注册回调实例
callback = MyXtQuantTraderCallback()
xt_trader.register_callback(callback)


# 3.xttrader连接miniQMT终端
xt_trader.start()
if xt_trader.connect() == 0:logging.info('【软件终端连接成功！】')
else: logging.info('【软件终端连接失败！】','\n 请运行并登录miniQMT.EXE终端。','\n path=改成你的QMT安装路径')           


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

