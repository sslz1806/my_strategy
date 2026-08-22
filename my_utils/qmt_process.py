import os
import subprocess
import time
import psutil
import win32com.client  # 需安装 pywin32
import win32con


def get_shortcut_target(shortcut_path):
    """
    解析 Windows 快捷方式（.lnk），获取指向的真实文件路径
    """
    if not shortcut_path.endswith(".lnk"):
        return None
    if not os.path.exists(shortcut_path):
        return None
    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        shortcut = shell.CreateShortCut(shortcut_path)
        return shortcut.Targetpath
    except Exception:
        return None


def get_process_name_from_path(software_path):
    """
    从软件路径（exe/lnk）中提取进程名
    """
    if not software_path:
        return None
    # 先解析快捷方式
    if software_path.endswith(".lnk"):
        software_path = get_shortcut_target(software_path)
    # 提取进程名（如 C:\\xxx\\XtItClient.exe → XtItClient.exe）
    if software_path and os.path.exists(software_path):
        return os.path.basename(software_path)
    return None


def is_process_running(process_name):
    """
    检测指定进程是否正在运行
    """
    if not process_name:
        return False
    for proc in psutil.process_iter(['name']):
        try:
            if proc.info['name'] and proc.info['name'].lower() == process_name.lower():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def start_software(
    software_path,
    avoid_duplicate=True,
    show_window=True,
    process_name=None,
    max_wait=30,
    poll_interval=1,
):
    """
    启动软件（兼容快捷方式/直接路径），修复仅后台运行的问题

    :param software_path: 软件路径（exe 或 lnk）
    :param avoid_duplicate: 是否避免重复启动（True=仅未运行时启动）
    :param show_window: 是否显示窗口（部分程序需要显示界面才能正常初始化）
    :param process_name: 真正需要等待的运行进程名。启动器与主进程不同时必须显式传入
    :param max_wait: 等待运行进程出现的最长秒数
    :param poll_interval: 轮询运行进程的间隔秒数
    :return: True=启动成功/已运行，False=启动失败
    """
    # 步骤1：解析快捷方式（如果是 lnk 文件）
    if software_path.endswith(".lnk"):
        software_path = get_shortcut_target(software_path)
        if not software_path:
            return False

    # 步骤2：检查真实路径是否存在
    if not os.path.exists(software_path):
        return False

    # 步骤3：如果需要避免重复启动，先检测进程
    process_name = process_name or get_process_name_from_path(software_path)
    if avoid_duplicate and process_name and is_process_running(process_name):
        return True  # 已运行也算「成功」（避免重复）

    # 步骤4：启动软件
    try:
        # 获取程序所在目录（关键：很多程序需要以自身目录为工作目录才能正常显示界面）
        work_dir = os.path.dirname(software_path)
        startup_info = subprocess.STARTUPINFO()
        if show_window:
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup_info.wShowWindow = win32con.SW_SHOWNORMAL  # 显示窗口

        # DETACHED_PROCESS 只脱离控制台，并不会脱离计划任务或 IDE 施加的
        # Windows Job。CREATE_BREAKAWAY_FROM_JOB 用于避免父任务结束时连带关闭 QMT。
        creation_flags = (
            subprocess.DETACHED_PROCESS
            | subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_BREAKAWAY_FROM_JOB
        )
        popen_kwargs = {
            "cwd": work_dir,
            "startupinfo": startup_info,
            "shell": False,
            "creationflags": creation_flags,
        }
        try:
            subprocess.Popen(software_path, **popen_kwargs)
        except OSError as exc:
            # 少数受限 Job 不允许子进程主动脱离。此时退回普通的独立进程启动，
            # 至少保证 QMT 可以拉起；无论哪条路径，本函数都不取得 QMT 的关闭权。
            if getattr(exc, "winerror", None) != 5:
                raise
            popen_kwargs["creationflags"] = (
                subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NEW_PROCESS_GROUP
            )
            subprocess.Popen(software_path, **popen_kwargs)

        # 启动器可能很快退出，因此必须等待调用方指定的真正运行进程。
        waited = 0
        while waited <= max_wait:
            if process_name and is_process_running(process_name):
                return True
            if waited >= max_wait:
                break
            sleep_seconds = min(poll_interval, max_wait - waited)
            time.sleep(sleep_seconds)
            waited += sleep_seconds

        # QMT 属于外部桌面程序。即使检测超时，也不能擅自终止已拉起的进程；
        # 它可能仍在登录、更新或由启动器切换到另一个子进程。
        return False
    except Exception:
        return False


def connect_trader_with_retry(
    trader_factory,
    ensure_client,
    max_attempts=7,
    retry_interval=4,
    logger=None,
):
    """
    确保 QMT 运行后创建并连接 trader，失败时重新检测 QMT 并创建新会话。

    XtQuantTrader.start() 只负责启动 xtquant 自身的异步线程，connect() 才会
    连接 miniQMT。启动阶段连接失败后复用同一个 trader 可能持续命中失效会话，
    因此每次重试都重新创建 trader；stop() 只清理该 trader 的线程，不关闭 QMT。

    :param trader_factory: 每次调用返回一个已注册回调、但尚未 start 的 trader
    :param ensure_client: 检测 QMT 主进程，未运行时负责拉起；成功返回 True
    :param max_attempts: 最大连接次数
    :param retry_interval: 相邻连接尝试之间的等待秒数
    :param logger: 可选日志对象
    :return: 已成功连接的 trader
    :raises ConnectionError: 所有尝试均失败
    """
    if max_attempts < 1:
        raise ValueError("max_attempts 必须大于等于 1")
    if retry_interval < 0:
        raise ValueError("retry_interval 不能小于 0")

    last_error = None
    for attempt in range(1, max_attempts + 1):
        trader = None
        try:
            if not ensure_client():
                raise ConnectionError("QMT 客户端未正常运行")

            # trader 必须在 QMT 主进程出现后创建，避免持有启动阶段的失效会话。
            trader = trader_factory()
            trader.start()
            connect_result = trader.connect()
            if connect_result == 0:
                if logger:
                    logger.info(f"第{attempt}/{max_attempts}次连接 miniQMT 成功")
                return trader
            last_error = ConnectionError(
                f"xtquant connect() 返回失败码: {connect_result}"
            )
        except Exception as exc:
            last_error = exc

        if trader is not None:
            try:
                trader.stop()
            except Exception as stop_error:
                if logger:
                    logger.warning(f"清理失败的 xtquant 会话时发生异常: {stop_error}")

        if logger:
            logger.warning(
                f"第{attempt}/{max_attempts}次连接 miniQMT 失败: {last_error}"
            )
        if attempt < max_attempts:
            time.sleep(retry_interval)

    raise ConnectionError(
        f"连续 {max_attempts} 次连接 miniQMT 失败，最后错误: {last_error}"
    ) from last_error


def stop_software(target, force=False, timeout=5):
    """
    检测并关闭软件（兼容进程名/EXE路径/快捷方式路径）

    :param target: 目标标识（可选：进程名/EXE路径/快捷方式.lnk路径）
    :param force: 是否强制终止（True=kill，False=terminate）
    :param timeout: 等待进程终止的最大秒数
    :return: 关闭成功的进程数（0=未运行/关闭失败，>0=成功关闭）
    """
    # 步骤1：统一转换为进程名
    process_name = None
    if "." in target and not os.path.exists(target):
        process_name = target
    elif os.path.exists(target):
        process_name = get_process_name_from_path(target)

    if not process_name:
        return 0

    # 步骤2：检测并关闭进程
    killed_count = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == process_name:
                pid = proc.info['pid']
                if force:
                    proc.kill()  # 强制终止
                else:
                    proc.terminate()  # 正常终止
                proc.wait(timeout=timeout)
                killed_count += 1
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            continue
        except psutil.TimeoutExpired:
            continue
        except Exception:
            continue

    return killed_count
