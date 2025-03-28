from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn import base
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder


# DataFrameのメモリ使用量を減らす関数
# 内容:より低い精度で表現できるカラムは精度を落とした型を使うように変更する
# 参考: https://qiita.com/kaggle_grandmaster-arai-san/items/d59b2fb7142ec7e270a5#reduce_mem_usage
def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのメモリ使用量を削減する関数

    各カラムのデータ範囲に応じて、より少ないメモリ使用量の型に変換します。
    整数型と浮動小数点型の両方に対応し、オブジェクト型はカテゴリ型に変換します。
    元のDataFrameは変更せず、変換後の新しいDataFrameを返します。

    Args:
        df: メモリ使用量を削減したいDataFrame

    Returns:
        メモリ使用量が削減されたDataFrame（新しいコピー）
    """
    # 元のDataFrameを変更しないようにコピーを作成
    result = df.copy()

    for col in result.columns:
        col_type = result[col].dtype

        if col_type is not object:
            c_min = result[col].min()
            c_max = result[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    result[col] = result[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    result[col] = result[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    result[col] = result[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    result[col] = result[col].astype(np.int64)
            else:
                # 浮動小数点型の場合
                try:
                    # float16の範囲チェックを安全に行う
                    f16_min = np.finfo(np.float16).min
                    f16_max = np.finfo(np.float16).max

                    if c_min > f16_min and c_max < f16_max:
                        result[col] = result[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        result[col] = result[col].astype(np.float32)
                    else:
                        result[col] = result[col].astype(np.float64)
                except (TypeError, ValueError):
                    # 変換できない場合は元の型を維持
                    pass
        else:
            result[col] = result[col].astype("category")

    return result


# カテゴリカルデータのみ抽出
def extracting_categorical_data(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameからカテゴリカル変数のみを抽出する関数

    Args:
        df: 元のDataFrame

    Returns:
        カテゴリカル変数のみを含むDataFrame
    """
    return df.select_dtypes(include=["object", "category"])


# 数値データのみ抽出
def extracting_number_data(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameから数値変数のみを抽出する関数

    Args:
        df: 元のDataFrame

    Returns:
        数値変数のみを含むDataFrame
    """
    return df.select_dtypes(include=["number"])


# Lable Encoding
# カテゴリカルデータの各カテゴリを整数に置き換える変換
def label_encode(df: pd.DataFrame, exclude_cols: List[str] = [], output_suffix: str = "") -> pd.DataFrame:
    """カテゴリカル変数にラベルエンコーディングを適用する関数

    カテゴリカルな各値を一意の整数値に変換します。

    Args:
        df: 変換対象のDataFrame
        exclude_cols: エンコーディングから除外するカラムのリスト
        output_suffix: 変換後のカラム名に付加する接尾辞

    Returns:
        ラベルエンコーディングを適用したDataFrame
    """
    result = df.copy()

    # カテゴリカルな列を抽出
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    target_cols = [col for col in categorical_cols if col not in exclude_cols]

    for col in target_cols:
        le = SklearnLabelEncoder()
        # 欠損値があるとエラーになるため対応
        is_null = result[col].isnull()
        non_null_values = result.loc[~is_null, col]

        if len(non_null_values) > 0:
            result.loc[~is_null, col + output_suffix] = le.fit_transform(non_null_values)
            result.loc[is_null, col + output_suffix] = np.nan

    return result


# カテゴリカルデータの組み合わせ
def combine_categorical_data(
    df: pd.DataFrame,
    exclude_cols: List[str] = [],
    output_suffix: str = "_re",
    r: int = 2,  # 組み合わせる変数の数
) -> pd.DataFrame:
    """カテゴリカルデータの組み合わせを生成する関数

    複数のカテゴリカル変数を組み合わせて新しいカテゴリカル変数を作成します。

    Args:
        df: 変換対象のDataFrame
        exclude_cols: 組み合わせから除外するカラムのリスト
        output_suffix: 変換後のカラム名に付加する接尾辞
        r: 組み合わせるカラムの数

    Returns:
        カテゴリカル変数の組み合わせを含むDataFrame
    """
    result = pd.DataFrame(index=df.index)

    # カテゴリカルな列を抽出
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    target_cols = [col for col in categorical_cols if col not in exclude_cols]

    # r個の列の組み合わせごとに新しい列を作成
    for cols in combinations(target_cols, r):
        new_col_name = "_".join(cols) + output_suffix
        # 文字列として連結
        result[new_col_name] = df[list(cols)].apply(lambda x: "_".join(x.astype(str)), axis=1)

    return result


# 指定した変数の数値データの加算
def additional_num_data(
    df: pd.DataFrame,
    input_cols: Optional[List[str]] = None,
    drop_origin: bool = True,
    operator: str = "+",
    r: int = 2,  # 組み合わせる変数の数
) -> pd.DataFrame:
    """数値変数の演算による特徴量を生成する関数

    指定された演算子を使用して、数値変数の組み合わせによる新たな特徴量を作成します。
    演算の結果、データ型の範囲を超える場合は自動的に適切なデータ型に変換されます。

    Args:
        df: 変換対象のDataFrame
        input_cols: 演算対象のカラムリスト（Noneの場合は全数値カラムを使用）
        drop_origin: Trueの場合は演算結果のみを返し、Falseの場合は元のDataFrame列も全て保持
        operator: 使用する演算子（"+"、"-"、"*"、"/"）
        r: 組み合わせるカラムの数

    Returns:
        数値変数の演算結果を含むDataFrame

    Raises:
        ValueError: サポートされていない演算子が指定された場合
    """
    # まずは結果用のDataFrameを準備
    if drop_origin:
        result = pd.DataFrame(index=df.index)
    else:
        # drop_origin=Falseの場合、元のDataFrameのすべての列をコピー
        result = df.copy()

    # 必要な数値列を特定
    numeric_cols = df.select_dtypes(include=["number"]).columns

    if input_cols is None:
        target_cols = numeric_cols
    else:
        target_cols = [col for col in input_cols if col in numeric_cols]

    # 演算子に応じた操作を定義
    operations = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
    }

    op_func = operations.get(operator)
    if op_func is None:
        raise ValueError(f"Unsupported operator: {operator}")

    # r個の列の組み合わせごとに計算
    for cols in combinations(target_cols, r):
        new_col_name = f"{operator.join(cols)}"

        # 最初の列の値を安全な型に変換して計算を開始
        # すべての演算子で型の範囲を超えることがありうるため、安全な型で開始
        first_col = df[cols[0]]
        if np.issubdtype(first_col.dtype, np.integer):
            # 整数型の場合、より大きな整数型に変換（int32）
            temp_result = first_col.astype(np.int32)
        else:
            # 浮動小数点型の場合、float64に変換
            temp_result = first_col.astype(np.float64)

        # 残りの列に対して演算を適用
        for col in cols[1:]:
            # 演算を実行
            temp_result = op_func(temp_result, df[col])

            # 型の範囲チェックと必要に応じたアップキャスト
            if np.issubdtype(temp_result.dtype, np.integer):
                # 整数型の場合、範囲チェックと適切なアップキャスト
                max_val = temp_result.max()
                min_val = temp_result.min()

                # int32範囲チェック
                if max_val > np.iinfo(np.int32).max or min_val < np.iinfo(np.int32).min:
                    temp_result = temp_result.astype(np.int64)

            # オーバーフローチェック - 浮動小数点型への変換が必要かどうか
            # 除算の場合は常に浮動小数点型に変換
            if operator == "/" or (
                temp_result.dtype == np.int64
                and (
                    temp_result.max() > np.iinfo(np.int64).max * 0.9 or temp_result.min() < np.iinfo(np.int64).min * 0.9
                )
            ):
                temp_result = temp_result.astype(np.float64)

        # 計算結果を結果DataFrameに追加
        result[new_col_name] = temp_result

    return result


# 指定した変数の値の集約
def aggregation_num_data(
    df: pd.DataFrame,
    group_key: str = "group_col",
    group_values: List[str] = [],
    agg_methods: List[str] = ["mean", "max"],
) -> pd.DataFrame:
    """グループごとの集計値を特徴量として生成する関数

    指定されたグループキーに基づいて、数値変数の集計値を計算し、特徴量として提供します。

    Args:
        df: 変換対象のDataFrame
        group_key: グループ化に使用するカラム名
        group_values: 集計対象のカラムリスト
        agg_methods: 適用する集計関数のリスト（"mean"、"max"、"min"、"std"など）

    Returns:
        集計値を含むDataFrame
    """
    result = pd.DataFrame(index=df.index)

    # グループごとにagg_methodsに指定された集計を行い、その結果をマッピング
    for val in group_values:
        for method in agg_methods:
            agg_result = df.groupby(group_key)[val].agg(method)
            new_col = f"{group_key}_{val}_{method}"
            result[new_col] = df[group_key].map(agg_result)

    return result


class KFoldTargetEncoderTrain(base.BaseEstimator, base.TransformerMixin):
    """
    参考notebook：
    https://www.kaggle.com/code/anuragbantu/target-encoding-beginner-s-guide
    """

    def __init__(
        self,
        colnames: str,
        targetName: str,
        n_fold: int = 5,
        verbosity: bool = False,
        discardOriginal_col: bool = False,
    ) -> None:
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KFoldTargetEncoderTrain":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(self.targetName, str)
        assert isinstance(self.colnames, str)
        assert self.colnames in X.columns
        assert self.targetName in X.columns
        mean_of_target = X[self.targetName].mean()
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=42)
        col_mean_name = self.colnames + "_" + "Kfold_Target_Enc"
        X[col_mean_name] = np.nan
        for tr_ind, val_ind in kf.split(X):
            X_tr, X_val = X.iloc[tr_ind], X.iloc[val_ind]
            X.loc[X.index[val_ind], col_mean_name] = X_val[self.colnames].map(
                X_tr.groupby(self.colnames)[self.targetName].mean()
            )
            X[col_mean_name] = X[col_mean_name].fillna(mean_of_target)

        if self.verbosity:
            encoded_feature = X[col_mean_name].values
            print(
                "Correlation between the new feature, {} and , {} is {}.".format(
                    col_mean_name,
                    self.targetName,
                    np.corrcoef(X[self.targetName].values, encoded_feature)[0][1],
                )
            )
        if self.discardOriginal_col:
            X = X.drop(self.targetName, axis=1)
        return X


class KFoldTargetEncoderTest(base.BaseEstimator, base.TransformerMixin):
    """
    参考notebook：
    https://www.kaggle.com/code/anuragbantu/target-encoding-beginner-s-guide
    """

    def __init__(self, train: pd.DataFrame, colNames: str, encodedName: str) -> None:
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "KFoldTargetEncoderTest":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # カテゴリごとのエンコード値を計算
        mean = self.train[[self.colNames, self.encodedName]].groupby(self.colNames).mean().reset_index()

        # 辞書形式に変換
        mapping_dict = dict(zip(mean[self.colNames], mean[self.encodedName]))

        # 新しい列を作成し、マッピングを適用
        X = X.copy()  # 元のデータフレームを変更しないようにコピー
        X[self.encodedName] = X[self.colNames].map(mapping_dict)

        return X
