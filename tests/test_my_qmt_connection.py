import importlib.util
import os
import sys
import types
import unittest
from unittest import mock


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MY_QMT_PATH = os.path.join(PROJECT_ROOT, "my_utils", "my_qmt.py")
sys.path.insert(0, PROJECT_ROOT)

from my_utils import qmt_process


class TestMyQmtConnectionWiring(unittest.TestCase):
    def test_import_tracks_real_qmt_process_and_recreates_failed_trader(self):
        """模块初始化应等待 XtMiniQmt，并在首次连接失败后创建新会话。"""
        traders = []

        class FakeTrader:
            def __init__(self, path, session_id):
                self.path = path
                self.session_id = session_id
                self.started = False
                self.stopped = False
                traders.append(self)

            def register_callback(self, callback):
                self.callback = callback

            def start(self):
                self.started = True

            def connect(self):
                return -1 if len(traders) == 1 else 0

            def stop(self):
                self.stopped = True

            def subscribe(self, account):
                return 0

        class FakeStockAccount:
            def __init__(self, account_id):
                self.account_id = account_id

        fake_xttrader = types.ModuleType("xtquant.xttrader")
        fake_xttrader.XtQuantTrader = FakeTrader
        fake_xttrader.XtQuantTraderCallback = object
        fake_xttype = types.ModuleType("xtquant.xttype")
        fake_xttype.StockAccount = FakeStockAccount
        fake_xtquant = types.ModuleType("xtquant")
        fake_xtquant.xtconstant = types.SimpleNamespace()
        fake_fun = types.ModuleType("my_utils.fun")
        fake_fun.get_logger = mock.Mock(return_value=mock.Mock())

        fake_modules = {
            "xtquant": fake_xtquant,
            "xtquant.xttrader": fake_xttrader,
            "xtquant.xttype": fake_xttype,
            "my_utils.fun": fake_fun,
        }
        module_name = "my_utils.my_qmt_connection_test"
        spec = importlib.util.spec_from_file_location(module_name, MY_QMT_PATH)
        module = importlib.util.module_from_spec(spec)

        with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(
            qmt_process,
            "is_process_running",
            return_value=False,
        ), mock.patch.object(
            qmt_process,
            "start_software",
            return_value=True,
        ) as start_software, mock.patch.object(qmt_process.time, "sleep"):
            spec.loader.exec_module(module)

        self.assertEqual(module.QMT_PROCESS_NAME, "XtMiniQmt.exe")
        self.assertEqual(len(traders), 2)
        self.assertTrue(traders[0].stopped)
        self.assertIs(module.xt_trader, traders[1])
        self.assertNotEqual(traders[0].session_id, traders[1].session_id)
        self.assertEqual(start_software.call_count, 2)
        for call in start_software.call_args_list:
            self.assertEqual(call.kwargs["process_name"], "XtMiniQmt.exe")


if __name__ == "__main__":
    unittest.main()
