import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# import seaborn as sns

img = cv2.imread("img1.jpeg")
print(img.shape)

rows = 8
cols = 5

src_points = np.float32([[124, 714], [572, 732], [130, 1414], [562, 1426]])
dst_points = np.float32(
    [
        [0, 0],
        [448, 0],
        [0, 712],
        [448, 712],
    ]
)
transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

transformed_image = cv2.warpPerspective(
    np.array(img),
    transform_matrix,
    (448, 712),
)
cv2.imwrite("trans.png", transformed_image)
transformed_image = transformed_image[
    51:654,
    22:420,
]

chunks = []
for row_img in np.array_split(transformed_image, rows, axis=0):
    for chunk in np.array_split(row_img, cols, axis=1):
        chunks.append(chunk)
print(len(chunks))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

df = pd.DataFrame()
fig = plt.figure(figsize=(8, 5))
# fig, axes = plt.subplots(8, 5, figsize=(8, 5))
for i, chunk in enumerate(chunks):
    chunk = chunk[4:71, 12:67]
    chunk = cv2.cvtColor(chunk, cv2.COLOR_RGB2GRAY)
    # chunk = cv2.equalizeHist(chunk)
    chunk = cv2.medianBlur(chunk, 3)
    save_path = output_dir / f"chunk_{i:02d}.png"
    cv2.imwrite(str(save_path), chunk)
    # sns.histplot(chunk.flatten(), bins=10, ax=axes[i])

    ax = fig.add_subplot(8, 5, i + 1)
    hist, bins, _ = ax.hist(chunk.flatten(), bins=30)
    plt.ylim(0, 600)
    # plt.xticks([])
    # plt.yticks([])
    # if i == 0 or i == 4 or i == 8 or i == 12 or i == 16 or i == 20 or i == 24 or i == 28:
    if i % 5 != 0:
        plt.yticks([])
    elif i <= 34:
        plt.xticks([])
    # hist, bins, _ = sns.hist(chunk.flatten(), bins=30)

    peak_indices = argrelextrema(hist, np.greater, order=6)[0]
    peak_positions = bins[peak_indices]
    peak_distances = np.diff(peak_positions)
    print("ピークの位置:", peak_positions)
    print("ピーク間の距離:", peak_distances)
    mean = chunk.mean()
    var = chunk.var()
    std = chunk.std()
    kurt = kurtosis(chunk.flatten())
    skewness = skew(chunk.flatten())
    series = pd.Series(
        [mean, var, std, kurt, skewness, peak_distances[0]],
        index=["mean", "var", "std", "kurt", "skewness", "peak_distance"],
    )
    df = df._append([series], ignore_index=True)
    # if i == 1:
    #     hist, bins, _ = ax.hist(chunk.flatten(), bins=30)
    #     peak_indices = argrelextrema(hist, np.greater, order=10)[0]
    #     peak_positions = bins[peak_indices]
    #     peak_distances = np.diff(peak_positions)
    #     print("ピークの位置:", peak_positions)
    #     print("ピーク間の距離:", peak_distances)

plt.savefig("hist.png")
plt.show()
df.to_csv("output.csv")
