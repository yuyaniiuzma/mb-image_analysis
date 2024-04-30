import cv2
import numpy as np
from pathlib import Path
import pandas as pd

img = cv2.imread("red-rooster-cock-side-view-abstract_1284-16627.jpg")
img = img[500:, 500:]

rows = 5
cols = 7

chunks = []
for row_img in np.array_split(img, rows, axis=0):
    for chunk in np.array_split(row_img, cols, axis=1):
        chunks.append(chunk)
print(len(chunks))

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

df = pd.DataFrame()
for i, chunk in enumerate(chunks):
    save_path = output_dir / f"chunk_{i:02d}.png"
    cv2.imwrite(str(save_path), chunk)
    mean = chunk.mean()
    var = chunk.var()
    std = chunk.std()
    series = pd.Series([mean, var, std], index=['mean', 'var', 'std'])
    df = df._append([series], ignore_index=True)
df.to_csv('output.csv')
