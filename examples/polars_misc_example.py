import sys
from pathlib import Path

import polars as pl

# ソースコードへのパスを追加
sys.path.append(str(Path(__file__).parent.parent))

# 自作モジュールをインポート
from src.ml.polars_misc import (
    additional_num_data,
    aggregation_num_data,
    combine_categorical_data,
    extracting_categorical_data,
    extracting_number_data,
    label_encode,
    target_encode,
)


def main():
    """polars_miscモジュールの使用例を示す関数"""
    print("Polars Miscellaneous Functions Examples")
    print("=" * 50)

    # サンプルデータの作成
    df = pl.DataFrame(
        {
            "id": range(1, 11),
            "category_1": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "category_2": ["X", "X", "Y", "Z", "X", "Y", "Z", "Y", "X", "Z"],
            "numeric_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "numeric_2": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "group_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
        }
    )

    print("\nサンプルデータ:")
    print(df)

    # カテゴリカルデータの抽出
    cat_df = extracting_categorical_data(df)
    print("\nカテゴリカルデータの抽出結果:")
    print(cat_df)

    # 数値データの抽出
    num_df = extracting_number_data(df)
    print("\n数値データの抽出結果:")
    print(num_df)

    # ラベルエンコーディング
    le_df = label_encode(df, exclude_cols=["id"], output_suffix="_le")
    print("\nラベルエンコーディング結果:")
    print(le_df)

    # ターゲットエンコーディング
    te_df = target_encode(
        df,
        n_splits=3,
        input_cols=["category_1", "category_2"],
        target_col="target",
        output_suffix="_te",
    )
    print("\nターゲットエンコーディング結果:")
    print(te_df)

    # カテゴリカルデータの組み合わせ
    comb_df = combine_categorical_data(
        df,
        exclude_cols=["id", "target"],
        output_suffix="_combined",
        r=2,
    )
    print("\nカテゴリカルデータの組み合わせ結果:")
    print(comb_df)

    # 数値データの加算
    add_df = additional_num_data(
        df,
        input_cols=["numeric_1", "numeric_2"],
        drop_origin=False,
        operator="+",
        r=2,
    )
    print("\n数値データの加算結果:")
    print(add_df)

    # グループ集約
    agg_df = aggregation_num_data(
        df,
        group_key="group_id",
        group_values=["numeric_1", "numeric_2"],
        agg_methods=["mean", "max", "min"],
    )
    print("\nグループ集約結果:")
    print(agg_df)


if __name__ == "__main__":
    main()
