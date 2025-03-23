from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize

# 日本語フォントを有効化
japanize()


def plot_importance(
    model: Any, feature_names: List[str], figsize: Tuple[int, int] = (12, 8), top_n: int = None
) -> plt.Figure:
    """
    特徴量の重要度を棒グラフでプロット

    パラメータ:
    -----------
    model : sklearn estimator
        feature_importance_またはcoef_属性を持つモデル
    feature_names : List[str]
        特徴量の名前
    figsize : Tuple[int, int]
        図のサイズ
    top_n : int, optional
        表示する上位n個の特徴量

    戻り値:
    -------
    matplotlib.figure.Figure
        プロットの図オブジェクト
    """
    try:
        importance = model.feature_importances_
    except AttributeError:
        try:
            importance = np.abs(model.coef_[0])
        except AttributeError:
            raise ValueError("モデルにfeature_importances_またはcoef_属性がありません")

    indices = np.argsort(importance)

    if top_n is not None:
        indices = indices[-top_n:]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(indices)), importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title("特徴量の重要度")
    ax.set_xlabel("重要度")
    fig.tight_layout()

    return fig
