import sys
from sklearn.cluster import KMeans
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_hex(color_rbg):
    hex = []
    for rgb in color_rbg:
        hex.append('#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2])))
    return hex




img_org = cv2.imread(sys.argv[1])

img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

img = cv2.resize(img_org, (250, 250))

r, g, b = cv2.split(img)
r = r.flatten()
g = g.flatten()
b = b.flatten()

img = img.reshape(img.shape[0]* img.shape[1], 3)

print("Starting clustering...")

palette = KMeans(n_clusters=10, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0)
palette.fit(img)

print("\nClustering done.")

colors = palette.cluster_centers_
labels = palette.labels_

colors = colors.astype(int)

hex_codes = rgb_to_hex(colors)


print(f'The color palette is :\n\t HEX : {hex_codes} \n\t RGB : {colors}')

f, axarr = plt.subplots(2,1) 

plt.xticks(np.arange(10), hex_codes)
plt.yticks([])
axarr[0].imshow(img_org)
axarr[0].axis('off')
axarr[1].imshow([
    [
     color for color in colors  
    ]
])

plt.show()
