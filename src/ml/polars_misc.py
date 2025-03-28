from itertools import combinations
from typing import List, Optional

import polars as pl
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder


# カテゴリカルデータのみ抽出
def extracting_categorical_data(df: pl.DataFrame) -> pl.DataFrame:
    """DataFrameからカテゴリカル変数と文字列変数のみを抽出する関数

    Args:
        df: 元のDataFrame

    Returns:
        カテゴリカル変数と文字列変数のみを含むDataFrame
    """
    schema = df.schema
    cat_cols = [col for col, dtype in schema.items() if dtype in [pl.Categorical, pl.Utf8]]
    return df.select(cat_cols)


# 数値データのみ抽出
def extracting_number_data(df: pl.DataFrame) -> pl.DataFrame:
    """DataFrameから数値変数のみを抽出する関数

    Args:
        df: 元のDataFrame

    Returns:
        数値変数のみを含むDataFrame
    """
    schema = df.schema
    num_cols = [
        col
        for col, dtype in schema.items()
        if dtype
        in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
    ]
    return df.select(num_cols)


# Label Encoding
def label_encode(df: pl.DataFrame, exclude_cols: List[str] = [], output_suffix: str = "") -> pl.DataFrame:
    """カテゴリカル変数にラベルエンコーディングを適用する関数

    カテゴリカルな各値を一意の整数値に変換します。

    Args:
        df: 変換対象のDataFrame
        exclude_cols: エンコーディングから除外するカラムのリスト
        output_suffix: 変換後のカラム名に付加する接尾辞

    Returns:
        ラベルエンコーディングを適用したDataFrame
    """
    result = df.clone()
    schema = df.schema
    cat_cols = [col for col, dtype in schema.items() if dtype in [pl.Categorical, pl.Utf8]]
    target_cols = [col for col in cat_cols if col not in exclude_cols]

    for col in target_cols:
        # 欠損値を除外したユニークな値を取得
        unique_values = df.filter(pl.col(col).is_not_null()).get_column(col).unique().to_list()

        if len(unique_values) > 0:
            # ラベルエンコーダを作成
            le = SklearnLabelEncoder()
            le.fit(unique_values)

            # マッピング辞書を作成
            mapping_dict = {val: i for i, val in enumerate(le.classes_)}

            # 新しいカラム名
            new_col = col + output_suffix

            # ラベルエンコーディングを適用（replace メソッドを使用）
            result = result.with_columns(
                pl.when(pl.col(col).is_null()).then(None).otherwise(pl.col(col).replace(mapping_dict)).alias(new_col)
            )

    return result


# Target Encoding
def target_encode(
    df: pl.DataFrame,
    n_splits: int = 5,
    input_cols: List[str] = [],
    target_col: str = "target",
    output_suffix: str = "_te",
) -> pl.DataFrame:
    """目的変数を使用してカテゴリカル変数をエンコーディングする関数

    Cross-Validationを用いて、各カテゴリに対応する目的変数の平均値でエンコーディングします。

    Args:
        df: 変換対象のDataFrame
        n_splits: Cross-Validationの分割数
        input_cols: エンコーディングする対象のカラムリスト
        target_col: 目的変数のカラム名
        output_suffix: 変換後のカラム名に付加する接尾辞

    Returns:
        ターゲットエンコーディングを適用したDataFrame
    """
    # 結果を格納するためのDataFrameを初期化
    result = pl.DataFrame({"index": range(len(df))})

    # Cross-Validationを行う
    fold = KFold(n_splits=n_splits, shuffle=False)

    # pandasに変換して処理（KFoldとの互換性のため）
    df_pd = df.to_pandas()

    # 全体の平均を計算
    global_mean = df_pd[target_col].mean()

    for col in input_cols:
        # 出力カラム名
        encoded_col = col + output_suffix
        result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(encoded_col))

        # 各foldでエンコーディングを実行
        for train_idx, test_idx in fold.split(df_pd):
            # 訓練データでのターゲットの平均を計算
            target_means = df_pd.iloc[train_idx].groupby(col)[target_col].mean().to_dict()

            # テストデータにエンコーディングを適用
            for idx in test_idx:
                value = df_pd.iloc[idx][col]
                encoded_value = target_means.get(value, global_mean)
                result = (
                    result.with_row_count("row_idx")
                    .with_columns(
                        pl.when(pl.col("row_idx") == idx)
                        .then(pl.lit(encoded_value))
                        .otherwise(pl.col(encoded_col))
                        .alias(encoded_col)
                    )
                    .drop("row_idx")
                )

    # インデックス列を削除
    result = result.drop("index")

    return result


# カテゴリカルデータの組み合わせ
def combine_categorical_data(
    df: pl.DataFrame,
    exclude_cols: List[str] = [],
    output_suffix: str = "_re",
    r: int = 2,  # 組み合わせる変数の数
) -> pl.DataFrame:
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
    # 結果用のDataFrameを初期化（元のDataFrameをコピーして使用）
    result = pl.DataFrame()

    # カテゴリカルな列を抽出
    schema = df.schema
    cat_cols = [col for col, dtype in schema.items() if dtype in [pl.Categorical, pl.Utf8]]
    target_cols = [col for col in cat_cols if col not in exclude_cols]

    # r個の列の組み合わせごとに新しい列を作成
    for cols in combinations(target_cols, r):
        new_col_name = "_".join(cols) + output_suffix

        # 文字列として連結する式を作成
        concat_expr = None
        for i, col in enumerate(cols):
            if i == 0:
                concat_expr = df[col].cast(pl.Utf8)
            else:
                concat_expr = concat_expr + "_" + df[col].cast(pl.Utf8)

        # 新しい列を追加
        result = result.with_columns(pl.Series(name=new_col_name, values=concat_expr))

    return result


# 指定した変数の数値データの加算
def additional_num_data(
    df: pl.DataFrame,
    input_cols: Optional[List[str]] = None,
    drop_origin: bool = True,
    operator: str = "+",
    r: int = 2,  # 組み合わせる変数の数
) -> pl.DataFrame:
    """数値変数の演算による特徴量を生成する関数

    指定された演算子を使用して、数値変数の組み合わせによる新たな特徴量を作成します。

    Args:
        df: 変換対象のDataFrame
        input_cols: 演算対象のカラムリスト（Noneの場合は全数値カラムを使用）
        drop_origin: Trueの場合、元のカラムを結果に含めない
        operator: 使用する演算子（"+"、"-"、"*"、"/"）
        r: 組み合わせるカラムの数

    Returns:
        数値変数の演算結果を含むDataFrame

    Raises:
        ValueError: サポートされていない演算子が指定された場合
    """
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

    # 結果のDataFrame
    if not drop_origin:
        result = df.clone()
    else:
        result = pl.DataFrame()

    # 数値カラムを抽出
    schema = df.schema
    numeric_cols = [
        col
        for col, dtype in schema.items()
        if dtype
        in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64]
    ]

    if input_cols is None:
        target_cols = numeric_cols
    else:
        target_cols = [col for col in input_cols if col in numeric_cols]

    # r個の列の組み合わせごとに計算
    for cols in combinations(target_cols, r):
        new_col_name = f"{operator.join(cols)}"

        # 初期値を設定
        value = df[cols[0]]

        # 残りの列に対して演算を適用
        for col in cols[1:]:
            if operator == "+":
                value = value + df[col]
            elif operator == "-":
                value = value - df[col]
            elif operator == "*":
                value = value * df[col]
            elif operator == "/":
                value = value / df[col]

        # 新しい列を追加
        result = result.with_columns(pl.Series(name=new_col_name, values=value))

    # 元のカラムを追加（drop_originがFalseの場合のみ）
    if not drop_origin and len(result.columns) == 0:
        result = df.clone()

    return result


class KFoldTargetEncoderTrain:
    """K分割交差検証を用いてターゲットエンコーディングを行うトレーニングデータ用トランスフォーマー

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
        """初期化

        Args:
            colnames: エンコードするカテゴリ変数のカラム名
            targetName: ターゲット変数のカラム名
            n_fold: 交差検証の分割数
            verbosity: True の場合、相関係数などの詳細情報を表示
            discardOriginal_col: True の場合、元のターゲット列を削除
        """
        self.colnames = colnames
        self.targetName = targetName
        self.n_fold = n_fold
        self.verbosity = verbosity
        self.discardOriginal_col = discardOriginal_col

    def fit(self, X: pl.DataFrame) -> "KFoldTargetEncoderTrain":
        """フィッティング関数 (このクラスでは何もしない)

        Args:
            X: 入力データフレーム

        Returns:
            自身のインスタンス
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """データ変換関数

        Args:
            X: 変換対象のデータフレーム (colnames と targetName を含む)

        Returns:
            エンコードされた新しい特徴量を含むデータフレーム
        """
        assert isinstance(self.targetName, str)
        assert isinstance(self.colnames, str)
        assert self.colnames in X.columns
        assert self.targetName in X.columns

        # ターゲットの全体平均を計算
        mean_of_target = X[self.targetName].mean()

        # K-Fold分割準備
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=42)

        # 新しいカラム名を設定
        col_mean_name = self.colnames + "_" + "Kfold_Target_Enc"

        # 結果を格納するためのDataFrameを初期化
        result = X.clone()
        result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col_mean_name))

        # インデックスとデータを用意 (KFold との互換性のため)
        X_pd = X.to_pandas()

        # 各foldでエンコーディングを実行
        for tr_ind, val_ind in kf.split(X_pd):
            # 訓練データでのカテゴリごとのターゲット平均を計算
            X_tr = X.filter(pl.lit(True).take_every(tr_ind))
            target_means = X_tr.group_by(self.colnames).agg(pl.col(self.targetName).mean().alias("mean")).collect()
            target_means_dict = {row[0]: row[1] for row in target_means.iter_rows()}

            # テストデータにエンコーディングを適用
            for idx in val_ind:
                value = X[idx, self.colnames]
                encoded_value = target_means_dict.get(value, mean_of_target)
                # 結果を更新
                result = (
                    result.with_row_count("row_idx")
                    .with_columns(
                        pl.when(pl.col("row_idx") == idx)
                        .then(pl.lit(encoded_value))
                        .otherwise(pl.col(col_mean_name))
                        .alias(col_mean_name)
                    )
                    .drop("row_idx")
                )

        # Verbosity が True の場合、相関係数を表示
        if self.verbosity:
            correlation = (
                pl.concat(
                    [pl.Series(X[self.targetName].to_numpy()), pl.Series(result[col_mean_name].to_numpy())],
                    how="horizontal",
                )
                .corr()
                .to_numpy()[0, 1]
            )
            print(f"相関係数 ({col_mean_name}, {self.targetName}) = {correlation}")

        # 元のカラムを削除する場合
        if self.discardOriginal_col:
            result = result.drop(self.targetName)

        return result


class KFoldTargetEncoderTest:
    """テストデータに対するターゲットエンコーディングを適用するトランスフォーマー

    参考notebook：
    https://www.kaggle.com/code/anuragbantu/target-encoding-beginner-s-guide
    """

    def __init__(self, train: pl.DataFrame, colNames: str, encodedName: str) -> None:
        """初期化

        Args:
            train: トレーニングデータのデータフレーム
            colNames: エンコードするカテゴリ変数のカラム名
            encodedName: エンコードされた特徴量の名前
        """
        self.train = train
        self.colNames = colNames
        self.encodedName = encodedName

    def fit(self, X: pl.DataFrame) -> "KFoldTargetEncoderTest":
        """フィッティング関数 (このクラスでは何もしない)

        Args:
            X: 入力データフレーム

        Returns:
            自身のインスタンス
        """
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """データ変換関数

        Args:
            X: 変換対象のデータフレーム

        Returns:
            エンコードされた新しい特徴量を含むデータフレーム
        """
        # カテゴリごとのエンコード値を計算
        mean = (
            self.train.select([self.colNames, self.encodedName])
            .group_by(self.colNames)
            .agg(pl.col(self.encodedName).mean())
        )

        # マッピング辞書を作成
        mapping_dict = {row[0]: row[1] for row in mean.iter_rows()}

        # 結果を格納するためのDataFrameを初期化
        result = X.clone()

        # 新しい列を作成し、マッピングを適用
        result = result.with_columns(pl.col(self.colNames).replace(mapping_dict, default=None).alias(self.encodedName))

        return result


# 指定した変数の値の集約
def aggregation_num_data(
    df: pl.DataFrame,
    group_key: str = "group_col",
    group_values: List[str] = [],
    agg_methods: List[str] = ["mean", "max"],
) -> pl.DataFrame:
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
    result = pl.DataFrame({"index": range(len(df))})

    # 集計方法をPolars表現に変換
    agg_method_map = {"mean": "mean", "max": "max", "min": "min", "std": "std", "count": "count", "sum": "sum"}

    # グループごとにagg_methodsに指定された集計を行う
    for val in group_values:
        for method in agg_methods:
            if method not in agg_method_map:
                continue

            # 集計を実行
            agg_expr = getattr(pl.col(val), agg_method_map[method])()
            agg_df = df.group_by(group_key).agg(agg_expr.alias("agg_value"))

            # 辞書にマッピング
            agg_dict = {row[0]: row[1] for row in agg_df.iter_rows()}

            # 新しいカラム名
            new_col = f"{group_key}_{val}_{method}"

            # マッピングを適用（replace メソッドを使用）
            mapping_dict = {i: agg_dict.get(df[i, group_key]) for i in range(len(df))}
            result = result.with_columns(pl.col("index").replace(mapping_dict, default=None).alias(new_col))

    # インデックス列を削除
    result = result.drop("index")

    return result
