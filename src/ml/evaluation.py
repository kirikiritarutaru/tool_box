from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from japanize_matplotlib import japanize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.multiclass import unique_labels

# 日本語フォントを有効化
japanize()


def plot_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    fontsize: Dict[str, int] = {"title": 20, "label": 12, "ticks": 10, "values": 12},
    colorbar: bool = True,
    rotation: int = 45,
) -> plt.Figure:
    """
    混同行列をプロット

    パラメータ:
    -----------
    y_true : numpy.ndarray
        真のラベル
    y_pred : numpy.ndarray
        予測ラベル
    class_names : List[str], optional
        クラス名のリスト。指定がない場合は一意のラベルから自動生成
    title : str, optional
        プロットのタイトル。指定がない場合は正規化に応じて自動設定
    figsize : Tuple[int, int], default=(10, 8)
        図のサイズ
    cmap : str, default="Blues"
        カラーマップ
    normalize : bool, default=False
        行ごとに正規化するかどうか
    fontsize : Dict[str, int]
        各要素のフォントサイズ設定
    colorbar : bool, default=True
        カラーバーを表示するかどうか
    rotation : int, default=45
        x軸ラベルの回転角度

    戻り値:
    -------
    matplotlib.figure.Figure
        プロットの図オブジェクト
    """
    # タイトルの設定
    if title is None:
        title = "正規化混同行列" if normalize else "混同行列"

    # 混同行列を計算
    cm = confusion_matrix(y_true, y_pred)

    # クラス名の設定
    labels = unique_labels(y_true, y_pred)
    if class_names is None:
        class_names = [str(i) for i in labels]
    else:
        # class_namesが配列の場合、適切なラベルのみを選択
        if hasattr(class_names, "__len__") and not isinstance(class_names, str) and len(class_names) > len(labels):
            try:
                class_names = [class_names[i] for i in labels]
            except (IndexError, TypeError):
                # インデックスアクセスできない場合はそのまま使用
                pass

    # 正規化処理
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # プロット作成
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.get_cmap(cmap))

    # カラーバーの設定
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.15)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=fontsize.get("ticks", 10))

    # 軸の設定
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="真のラベル",
        xlabel="予測ラベル",
    )

    # フォントサイズの設定
    ax.set_title(title, fontsize=fontsize.get("title", 20))
    ax.set_xlabel("予測ラベル", fontsize=fontsize.get("label", 12))
    ax.set_ylabel("真のラベル", fontsize=fontsize.get("label", 12))
    plt.setp(
        ax.get_xticklabels(), rotation=rotation, ha="right", rotation_mode="anchor", fontsize=fontsize.get("ticks", 10)
    )
    plt.setp(ax.get_yticklabels(), fontsize=fontsize.get("ticks", 10))

    # テキストとして値を表示
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                fontsize=fontsize.get("values", 12),
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    return fig


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    分類モデルを評価し、様々な指標を計算

    パラメータ:
    -----------
    y_true : numpy.ndarray
        真のラベル
    y_pred : numpy.ndarray
        予測ラベル
    y_prob : numpy.ndarray, optional
        確率予測（ROC AUCなどに必要）

    戻り値:
    -------
    Dict[str, Any]
        各評価指標を含む辞書
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "classification_report": classification_report(y_true, y_pred),
    }

    # 確率予測がある場合はROC AUCも計算
    if y_prob is not None and len(np.unique(y_true)) == 2:
        results["roc_auc"] = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        results["roc_curve"] = (fpr, tpr)

    return results


def plot_roc(
    y_true: np.ndarray, y_prob: np.ndarray, title: str = "ROC曲線", figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    ROC曲線をプロット

    パラメータ:
    -----------
    y_true : numpy.ndarray
        真のラベル
    y_prob : numpy.ndarray
        確率予測
    title : str
        プロットのタイトル
    figsize : Tuple[int, int]
        図のサイズ

    戻り値:
    -------
    matplotlib.figure.Figure
        プロットの図オブジェクト
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, lw=2, label=f"ROC曲線 (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("偽陽性率")
    ax.set_ylabel("真陽性率")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)

    fig.tight_layout()
    return fig
