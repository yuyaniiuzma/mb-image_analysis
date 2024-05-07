import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt

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
    chunk = chunk[2:77, 9:71]
    chunk = cv2.cvtColor(chunk, cv2.COLOR_RGB2GRAY)
    save_path = output_dir / f"chunk_{i:02d}.png"
    cv2.imwrite(str(save_path), chunk)
    # sns.histplot(chunk.flatten(), bins=10, ax=axes[i])
    ax = fig.add_subplot(8, 5, i + 1)
    ax.hist(chunk.flatten(), bins=30)
    mean = chunk.mean()
    var = chunk.var()
    std = chunk.std()
    kurt = kurtosis(chunk.flatten())
    skewness = skew(chunk.flatten())
    series = pd.Series(
        [mean, var, std, kurt, skewness],
        index=["mean", "var", "std", "kurt", "skewness"],
    )
    df = df._append([series], ignore_index=True)

plt.show()
df.to_csv("output.csv")
