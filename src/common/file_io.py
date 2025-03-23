import pathlib
from typing import Dict, List, Optional, Union


def make_dir(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    指定されたパスにディレクトリを作成

    パラメータ:
    -----------
    path : str または pathlib.Path
        作成するディレクトリのパス

    戻り値:
    -------
    pathlib.Path
        作成されたディレクトリのパス
    """
    dir_path = pathlib.Path(path)
    dir_path.mkdir(exist_ok=True, parents=True)
    return dir_path


def make_dirs(
    base_path: Union[str, pathlib.Path], subdirs: List[str] = ["data", "models", "reports"]
) -> Dict[str, pathlib.Path]:
    """
    プロジェクトのディレクトリ構造を作成

    パラメータ:
    -----------
    base_path : str または pathlib.Path
        ベースディレクトリのパス
    subdirs : List[str]
        作成するサブディレクトリのリスト

    戻り値:
    -------
    Dict[str, pathlib.Path]
        各ディレクトリ名をキー、パスを値とする辞書

    使用例:
    -------
    >>> dirs = make_dirs("/path/to/project")
    >>> print(dirs)
    {'data': PosixPath('/path/to/project/data'), 'models': PosixPath('/path/to/project/models'), 'reports': PosixPath('/path/to/project/reports')}

    >>> dirs = make_dirs("/path/to/project", ["raw_data", "processed_data", "output"])
    >>> print(dirs["raw_data"])
    PosixPath('/path/to/project/raw_data')
    """
    base_path = pathlib.Path(base_path)
    dirs = {}

    for subdir in subdirs:
        dir_path = base_path / subdir
        dir_path.mkdir(exist_ok=True, parents=True)
        dirs[subdir] = dir_path

    return dirs


def get_files(
    directory: Union[str, pathlib.Path], extensions: Optional[List[str]] = None, recursive: bool = False
) -> List[pathlib.Path]:
    """
    指定ディレクトリ内のファイルリストを取得

    パラメータ:
    -----------
    directory : str または pathlib.Path
        検索対象のディレクトリ
    extensions : Optional[List[str]]
        フィルタリングする拡張子のリスト (例: ['.txt', '.csv'])
        None の場合は全てのファイルを返す
    recursive : bool
        True の場合、サブディレクトリも再帰的に検索

    戻り値:
    -------
    List[pathlib.Path]
        見つかったファイルのパスのリスト
    """
    directory = pathlib.Path(directory)

    if not recursive:
        files = [f for f in directory.iterdir() if f.is_file()]
    else:
        files = [f for f in directory.glob("**/*") if f.is_file()]

    if extensions:
        # 拡張子のリストが指定されている場合、フィルタリング
        # 拡張子の先頭のドットがない場合は自動追加
        normalized_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        files = [f for f in files if f.suffix.lower() in normalized_extensions]

    return files


def normalize_path(path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    パスを正規化する

    パラメータ:
    -----------
    path : str または pathlib.Path
        正規化するパス

    戻り値:
    -------
    pathlib.Path
        正規化されたパス
    """
    return pathlib.Path(path).expanduser().resolve()


def get_extension(path: Union[str, pathlib.Path]) -> str:
    """
    ファイルパスから拡張子を抽出する

    パラメータ:
    -----------
    path : str または pathlib.Path
        拡張子を抽出するファイルパス

    戻り値:
    -------
    str
        拡張子（ドット付き）。拡張子がない場合は空文字列
    """
    return pathlib.Path(path).suffix
