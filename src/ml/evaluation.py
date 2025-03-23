#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
プロットユーティリティモジュール
------------------------------
機械学習およびコンピュータビジョンプロジェクトで使用する
データの可視化関連の関数を提供します。
"""

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
from sklearn.model_selection import learning_curve
from sklearn.utils.multiclass import unique_labels

# 日本語フォントを有効化
japanize()


def plot_cm(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = None,
    title: str = "混同行列",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False,
) -> plt.Figure:
    """
    混同行列をプロット (旧名: plot_confusion_matrix)

    パラメータ:
    -----------
    y_true : numpy.ndarray
        真のラベル
    y_pred : numpy.ndarray
        予測ラベル
    class_names : List[str], optional
        クラス名のリスト
    title : str
        プロットのタイトル
    figsize : Tuple[int, int]
        図のサイズ
    cmap : str
        カラーマップ
    normalize : bool
        行ごとに正規化するかどうか

    戻り値:
    -------
    matplotlib.figure.Figure
        プロットの図オブジェクト
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # クラス名の設定
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

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

    # 軸ラベルを回転
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # テキストとして値を表示
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black"
            )

    fig.tight_layout()
    return fig


def plot_cm_enhanced(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    拡張版の混同行列プロット関数 (旧名: plot_confusion_matrix_enhanced)

    scikit-learnの例に基づいて実装された、より詳細な混同行列の可視化

    パラメータ:
    -----------
    y_true : array-like
        真のラベル
    y_pred : array-like
        予測ラベル
    classes : array-like
        クラス名のリスト
    normalize : bool, default=False
        正規化するかどうか
    title : str, optional
        プロットのタイトル
    cmap : matplotlib.colors.Colormap, default=plt.cm.Blues
        カラーマップ

    戻り値:
    -------
    matplotlib.axes.Axes
        プロットの軸オブジェクト
    """
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # 混同行列を計算
    cm = confusion_matrix(y_true, y_pred)
    # データに現れるラベルのみを使用
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, fontsize=12)
    plt.yticks(tick_marks, fontsize=12)
    plt.xlabel("Predicted label", fontsize=25)
    plt.ylabel("True label", fontsize=25)
    plt.title(title, fontsize=30)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = ax.figure.colorbar(im, ax=ax, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # 軸ラベルの回転と配置
    plt.setp(ax.get_xticklabels(), ha="center", rotation_mode="anchor")

    # データの次元ごとにテキストアノテーションを作成
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                fontsize=20,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    return ax


def plot_learning(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    train_sizes: np.ndarray = np.linspace(0.1, 1.0, 5),
    title: str = "学習曲線",
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    学習曲線をプロット (旧名: plot_learning_curve)

    パラメータ:
    -----------
    estimator : sklearn estimator
        モデル
    X : numpy.ndarray
        特徴量
    y : numpy.ndarray
        ターゲット
    cv : int
        交差検証の分割数
    train_sizes : numpy.ndarray
        トレーニングサイズの割合
    title : str
        プロットのタイトル
    figsize : Tuple[int, int]
        図のサイズ

    戻り値:
    -------
    matplotlib.figure.Figure
        プロットの図オブジェクト
    """
    fig, ax = plt.subplots(figsize=figsize)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring="accuracy"
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # トレーニングスコアとテストスコアをプロット
    ax.fill_between(
        train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r"
    )
    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g"
    )
    ax.plot(train_sizes, train_scores_mean, "o-", color="r", label="トレーニングスコア")
    ax.plot(train_sizes, test_scores_mean, "o-", color="g", label="交差検証スコア")

    ax.set_title(title)
    ax.set_xlabel("トレーニングサンプル数")
    ax.set_ylabel("スコア")
    ax.legend(loc="best")
    ax.grid(True)

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
