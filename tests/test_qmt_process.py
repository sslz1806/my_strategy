import os
import inspect
import sys
import unittest
from unittest import mock

# 把项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from my_utils import qmt_process
from my_utils.qmt_process import get_process_name_from_path, is_process_running


class TestQmtProcess(unittest.TestCase):
    def test_get_process_name_from_path(self):
        """测试从路径提取进程名（使用真实存在的 Python 解释器路径）"""
        python_exe = sys.executable
        self.assertEqual(get_process_name_from_path(python_exe), os.path.basename(python_exe))
        self.assertIsNone(get_process_name_from_path(r"C:\\a\\not_exist.exe"))
        self.assertIsNone(get_process_name_from_path(None))
        self.assertIsNone(get_process_name_from_path(""))

    def test_is_process_running_python(self):
        """测试当前 Python 进程一定能被检测到"""
        self.assertTrue(is_process_running("python.exe"))

    def test_start_software_waits_for_runtime_process_and_detaches_from_parent_job(self):
        """启动器退出后，应等待真正的 QMT 进程，并使其脱离父进程的 Job。"""
        self.assertIn(
            "process_name",
            inspect.signature(qmt_process.start_software).parameters,
            "启动接口必须能区分启动器和真正的运行进程",
        )
        popen = mock.Mock()
        running_states = iter([False, False, True])

        with mock.patch.object(
            qmt_process,
            "is_process_running",
            side_effect=lambda process_name: next(running_states),
        ) as process_checker, mock.patch.object(
            qmt_process.subprocess,
            "Popen",
            return_value=popen,
        ) as popen_factory, mock.patch.object(qmt_process.time, "sleep"):
            started = qmt_process.start_software(
                sys.executable,
                process_name="XtMiniQmt.exe",
                max_wait=2,
                poll_interval=1,
            )

        self.assertTrue(started)
        self.assertEqual(
            [call.args[0] for call in process_checker.call_args_list],
            ["XtMiniQmt.exe", "XtMiniQmt.exe", "XtMiniQmt.exe"],
        )
        creationflags = popen_factory.call_args.kwargs["creationflags"]
        self.assertTrue(creationflags & qmt_process.subprocess.DETACHED_PROCESS)
        self.assertTrue(creationflags & qmt_process.subprocess.CREATE_NEW_PROCESS_GROUP)
        self.assertTrue(creationflags & qmt_process.subprocess.CREATE_BREAKAWAY_FROM_JOB)

    def test_start_software_timeout_never_terminates_launched_qmt(self):
        """未及时识别到运行进程时，也不能擅自关闭已拉起的外部 QMT。"""
        self.assertIn(
            "process_name",
            inspect.signature(qmt_process.start_software).parameters,
            "启动接口必须能跟踪真正的 QMT 运行进程",
        )
        popen = mock.Mock()

        with mock.patch.object(
            qmt_process,
            "is_process_running",
            return_value=False,
        ), mock.patch.object(
            qmt_process.subprocess,
            "Popen",
            return_value=popen,
        ), mock.patch.object(qmt_process.time, "sleep"):
            started = qmt_process.start_software(
                sys.executable,
                process_name="XtMiniQmt.exe",
                max_wait=1,
                poll_interval=1,
            )

        self.assertFalse(started)
        popen.terminate.assert_not_called()

    def test_connect_trader_rechecks_qmt_and_recreates_trader_after_failure(self):
        """连接失败后，应重新检查 QMT，并用新 trader 会话重试。"""
        connect_with_retry = getattr(qmt_process, "connect_trader_with_retry", None)
        self.assertIsNotNone(connect_with_retry, "缺少 QMT 连接重试接口")

        failed_trader = mock.Mock()
        failed_trader.connect.return_value = -1
        connected_trader = mock.Mock()
        connected_trader.connect.return_value = 0
        trader_factory = mock.Mock(side_effect=[failed_trader, connected_trader])
        ensure_client = mock.Mock(return_value=True)

        with mock.patch.object(qmt_process.time, "sleep"):
            result = connect_with_retry(
                trader_factory,
                ensure_client,
                max_attempts=2,
                retry_interval=1,
            )

        self.assertIs(result, connected_trader)
        self.assertEqual(ensure_client.call_count, 2)
        failed_trader.start.assert_called_once_with()
        failed_trader.stop.assert_called_once_with()
        connected_trader.start.assert_called_once_with()
        connected_trader.stop.assert_not_called()

    def test_connect_trader_stops_failed_session_and_raises_after_exhaustion(self):
        """所有连接尝试失败时，应停止 xtquant 线程并明确报错，不能继续订阅账户。"""
        connect_with_retry = getattr(qmt_process, "connect_trader_with_retry", None)
        self.assertIsNotNone(connect_with_retry, "缺少 QMT 连接重试接口")

        trader = mock.Mock()
        trader.connect.return_value = -1

        with mock.patch.object(qmt_process.time, "sleep"):
            with self.assertRaises(ConnectionError):
                connect_with_retry(
                    mock.Mock(return_value=trader),
                    mock.Mock(return_value=True),
                    max_attempts=1,
                    retry_interval=0,
                )

        trader.stop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
