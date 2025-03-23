import pathlib
from typing import List, Optional, Union

import cv2
import numpy as np
from japanize_matplotlib import japanize

# 日本語フォントを有効化
japanize()


def save_image(image: np.ndarray, file_path: Union[str, pathlib.Path]) -> None:
    """
    画像を保存

    パラメータ:
    -----------
    image : numpy.ndarray
        保存する画像（OpenCV形式のBGR画像を想定）
    file_path : str または pathlib.Path
        保存先のパス
    """
    img_to_save = image.copy()

    # グレースケールの場合
    if len(img_to_save.shape) == 2 or (len(img_to_save.shape) == 3 and img_to_save.shape[2] == 1):
        if len(img_to_save.shape) == 3:
            img_to_save = img_to_save.squeeze()
        cv2.imwrite(str(file_path), img_to_save)
    else:
        # すでにBGR形式と想定
        cv2.imwrite(str(file_path), img_to_save)


def plot_images(
    images: np.ndarray,
    rows: int = 4,
    cols: int = 5,
    labels: Optional[List[str]] = None,
    window_name: str = "Images Grid",
    wait_key: int = 0,
) -> np.ndarray:
    """
    画像のグリッドを連結して表示

    パラメータ:
    -----------
    images : numpy.ndarray
        画像の配列 (N, H, W, C)
    labels : List[str], optional
        各画像のラベル
    rows : int
        行数
    cols : int
        列数
    window_name : str
        表示ウィンドウの名前
    wait_key : int
        cv2.waitKey()に渡す値。0の場合はキー入力待ち

    戻り値:
    -------
    numpy.ndarray
        連結された画像
    """
    n_images = min(rows * cols, len(images))

    # 画像サイズを統一（最初の画像のサイズを使用）
    if n_images > 0:
        h, w = images[0].shape[:2]
        canvas_h, canvas_w = h * rows, w * cols

        # キャンバス作成（背景は黒）
        if len(images[0].shape) == 3 and images[0].shape[2] == 3:
            # カラー画像
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        else:
            # グレースケール画像
            canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

        # 画像を配置
        for i in range(n_images):
            r, c = i // cols, i % cols
            img = images[i].copy()

            # リサイズが必要な場合
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))

            # グレースケールをカラーに変換する必要がある場合
            if len(canvas.shape) == 3 and len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            # カラーをグレースケールに変換する必要がある場合
            if len(canvas.shape) == 2 and len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            y_start, y_end = r * h, (r + 1) * h
            x_start, x_end = c * w, (c + 1) * w
            canvas[y_start:y_end, x_start:x_end] = img

            # ラベルを追加（視認性向上のため、太い文字の上に細い文字を重ねる）
            if labels is not None and i < len(labels):
                label_pos = (x_start + 5, y_start + 20)

                # 太い文字（縁取り効果）
                cv2.putText(canvas, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

                # 細い文字（内側）
                cv2.putText(
                    canvas, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

        # 画像表示
        cv2.imshow(window_name, canvas)
        cv2.waitKey(wait_key)

        return canvas

    return np.zeros((1, 1), dtype=np.uint8)  # 空の画像


def plot_image_comparison(
    images: List[np.ndarray],
    titles: List[str],
    window_name: str = "Image Comparison",
    wait_key: int = 0,
    is_bgr: bool = True,
) -> np.ndarray:
    """
    複数の画像を横に並べて比較表示

    パラメータ:
    -----------
    images : List[numpy.ndarray]
        比較する画像のリスト
    titles : List[str]
        各画像のタイトル
    window_name : str
        表示ウィンドウの名前
    wait_key : int
        cv2.waitKey()に渡す値。0の場合はキー入力待ち
    is_bgr : bool
        画像がOpenCV形式（BGR順）かどうか

    戻り値:
    -------
    numpy.ndarray
        連結された画像
    """
    n = len(images)

    if n == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    # 画像サイズを統一（最大の高さを使用）
    max_h = max(img.shape[0] for img in images)

    # 各画像の幅を計算
    widths = [img.shape[1] for img in images]
    total_width = sum(widths)

    # キャンバス作成
    if len(images[0].shape) == 3 and images[0].shape[2] == 3:
        # カラー画像
        canvas = np.zeros((max_h, total_width, 3), dtype=np.uint8)
    else:
        # グレースケール画像
        canvas = np.zeros((max_h, total_width), dtype=np.uint8)

    # 画像を配置
    x_offset = 0
    for i, (img, title) in enumerate(zip(images, titles)):
        h, w = img.shape[:2]

        # リサイズが必要な場合（高さのみ合わせる）
        if h != max_h:
            aspect_ratio = w / h
            new_w = int(max_h * aspect_ratio)
            img = cv2.resize(img, (new_w, max_h))
            h, w = img.shape[:2]

        # BGRからRGBに変換が必要な場合
        if is_bgr and len(img.shape) == 3 and img.shape[2] == 3:
            img = img.copy()  # 元の画像を変更しない

        # グレースケールをカラーに変換する必要がある場合
        if len(canvas.shape) == 3 and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # カラーをグレースケールに変換する必要がある場合
        if len(canvas.shape) == 2 and len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 画像を配置
        canvas[0:h, x_offset : x_offset + w] = img

        # タイトルを追加
        title_pos = (x_offset + 5, 20)

        # 太い文字（縁取り効果）
        cv2.putText(canvas, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # 細い文字（内側）
        cv2.putText(canvas, title, title_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 次の画像のx位置を更新
        x_offset += w

    # 画像表示
    cv2.imshow(window_name, canvas)
    cv2.waitKey(wait_key)

    return canvas
