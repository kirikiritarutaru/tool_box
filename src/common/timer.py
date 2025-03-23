from functools import wraps
from time import perf_counter


def time_msec(func):
    """
    関数の実行時間をミリ秒単位で計測するデコレータ

    使用例:
    -------
    >>> @time_msec
    >>> def process_data():
    >>>     # データ処理
    """

    @wraps(func)
    def wrapper(*args, **kargs):
        t0 = perf_counter()
        result = func(*args, **kargs)
        print(f"[{func.__name__}] done in {(perf_counter() - t0) * 1000:.0f} msec")
        return result

    return wrapper


def time_sec(func):
    """
    関数の実行時間を秒単位で計測するデコレータ

    使用例:
    -------
    >>> @time_sec
    >>> def process_data():
    >>>     # データ処理
    """

    @wraps(func)
    def wrapper(*args, **kargs):
        t0 = perf_counter()
        result = func(*args, **kargs)
        print(f"[{func.__name__}] done in {(perf_counter() - t0):.1f} sec")
        return result

    return wrapper


class Timer:
    """
    シンプルなタイマークラス

    使用例:
    -------
    >>> t = Timer("処理")
    >>> t.start()
    >>> # 何らかの処理
    >>> t.stop()  # 経過時間が表示される
    >>>
    >>> # または経過時間を取得
    >>> t = Timer().start()
    >>> # 処理
    >>> elapsed = t.elapsed()
    >>> print(f"経過: {elapsed:.2f}秒")
    """

    def __init__(self, name: str = "処理"):
        self.name = name
        self.start_time = None
        self.end_time = None

    def start(self):
        """タイマー開始"""
        self.start_time = perf_counter()
        self.end_time = None
        return self

    def stop(self) -> float:
        """タイマー停止"""
        if self.start_time is None:
            raise RuntimeError("タイマーが開始されていません")

        self.end_time = perf_counter()
        elapsed = self.end_time - self.start_time
        print(f"{self.name}: {elapsed:.2f}秒")
        return elapsed

    def elapsed(self) -> float:
        """経過時間を取得"""
        if self.start_time is None:
            raise RuntimeError("タイマーが開始されていません")

        if self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return perf_counter() - self.start_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
