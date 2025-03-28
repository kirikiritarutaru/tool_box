import sys
from pathlib import Path

import pandas as pd

# モジュールのインポートパスを追加（pathlibを使用）
sys.path.append(str(Path(__file__).parent.parent.absolute()))

# pandas_miscモジュールをインポート
from src.ml.pandas_misc import (
    KFoldTargetEncoderTest,
    KFoldTargetEncoderTrain,
    additional_num_data,
    aggregation_num_data,
    combine_categorical_data,
    extracting_categorical_data,
    extracting_number_data,
    label_encode,
    reduce_mem_usage,
)


def main():
    # サンプルデータの作成
    print("サンプルデータの作成...")

    # サンプルデータの作成
    df = pd.DataFrame(
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

    print("データサンプル:")
    print(df.head())
    print("\nデータ情報:")
    print(df.info())

    # メモリ使用量削減の例
    print("\n\n## メモリ使用量削減の例")
    # メモリ削減前のサイズを確認
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"削減前のメモリ使用量: {start_mem:.4f} MB")

    # メモリ削減処理を実行
    df_reduced = reduce_mem_usage(df)

    # メモリ削減後のサイズを確認
    end_mem = df_reduced.memory_usage(deep=True).sum() / 1024**2
    print(f"削減後のメモリ使用量: {end_mem:.4f} MB")
    print(f"削減率: {100 * (start_mem - end_mem) / start_mem:.2f}%")

    # データ型の変化を確認
    print("\n元データとメモリ削減後のデータ型比較:")
    df_dtypes = pd.DataFrame({"元データ型": df.dtypes, "削減後データ型": df_reduced.dtypes})
    print(df_dtypes)

    # カテゴリカルデータ抽出の例
    print("\n\n## カテゴリカルデータ抽出の例")
    cat_data = extracting_categorical_data(df)
    print(cat_data.head())

    # 数値データ抽出の例
    print("\n\n## 数値データ抽出の例")
    num_data = extracting_number_data(df)
    print(num_data.head())

    # ラベルエンコーディングの例
    print("\n\n## ラベルエンコーディングの例")
    df_encoded = label_encode(df, output_suffix="_le")
    print(df_encoded.head())

    # カテゴリカルデータの組み合わせの例
    print("\n\n## カテゴリカルデータの組み合わせの例")
    # category_1とcategory_2の組み合わせを作成
    combined_cats = combine_categorical_data(df, exclude_cols=["id"], output_suffix="_combined", r=2)
    print("カテゴリ組み合わせ結果:")
    print(combined_cats.head())

    # 数値データの演算例
    print("\n\n## 数値データの演算例")

    # 加算
    add_result = additional_num_data(df, input_cols=["numeric_1", "numeric_2"], drop_origin=False, operator="+", r=2)
    print("数値データ加算結果:")
    print(add_result.head())

    # 乗算
    mult_result = additional_num_data(df, input_cols=["numeric_1", "numeric_2"], drop_origin=True, operator="*", r=2)
    print("\n数値データ乗算結果:")
    print(mult_result.head())

    # グループごとの集計の例
    print("\n\n## グループごとの集計の例")
    # グループごとの数値カラムの平均と最大値を計算
    agg_result = aggregation_num_data(
        df,
        group_key="group_id",
        group_values=["numeric_1", "numeric_2"],
        agg_methods=["mean", "max", "min"],
    )

    print("集計結果:")
    print(agg_result.head())

    # 元データと結合して確認
    df_with_agg = pd.concat([df, agg_result], axis=1)
    print("\n元データとの結合結果:")
    print(df_with_agg.head())

    # グループごとの統計を確認
    print("\n各グループの統計情報（確認用）:")
    group_stats = df.groupby("group_id").agg(
        {
            "numeric_1": ["mean", "max", "min"],
            "numeric_2": ["mean", "max", "min"],
        }
    )
    print(group_stats)


def check_te():  # DataFrameで仮の訓練データセットを作成（10×3）
    df = pd.DataFrame(
        {
            "column1": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "column2": ["C", "D", "D", "D", "D", "D", "D", "D", "E", "E"],
            "target": [1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        }
    )

    targetc = KFoldTargetEncoderTrain("column2", "target", n_fold=3)
    new_train = targetc.fit_transform(df)
    print(new_train)

    test_targetc = KFoldTargetEncoderTest(new_train, "column2", "column2_Kfold_Target_Enc")
    new_test = test_targetc.fit_transform(df)
    print(new_test)


if __name__ == "__main__":
    # main()
    check_te()
