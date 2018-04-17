import os
import csv
import argparse

import cv2
import numpy as np
from sklearn.cluster import KMeans


# fetching jpg files
def img_file_paths(dir_path):
    file_paths = []
    for dirpath, _, filenames in os.walk(dir_path):
        for f in filenames:
            if f.endswith('.jpg') or f.endswith('.JPG'):
                file_paths.append(os.path.abspath(os.path.join(dirpath, f)))
    return file_paths


# masking relevant pixels
def process(file_paths):
    gcp_positions = []
    for file_path in file_paths:
        img = cv2.imread(file_path)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, (50, 0, 150), (130, 50, 255))

        # slice the white
        imask = mask > 0
        white = np.zeros_like(img, np.uint8)
        white[imask] = img[imask]

        wx, wy, wz = np.where(white > 0)

        if len(wx) >= 200:
            cluster = find_clusters(wx, wy, wz)
            data = {'Filename': file_path.split('/')[-1], 'GCP_position': cluster}
            gcp_positions.append(data)

    return gcp_positions


# Clustering pixels based on indices
def find_clusters(x, y, z):
    k = []
    for i in range(len(x)):
        k.append((x[i], y[i]))

    kmeans = KMeans(n_clusters=5)
    kmeans = kmeans.fit(k)
    labels = kmeans.predict(k)
    centroids = kmeans.cluster_centers_
    counts = np.bincount(labels)
    most_probable_cluster = np.argmax(counts)
    return centroids[most_probable_cluster]


def main():
    file_paths = img_file_paths('input')
    d = process(file_paths)
    keys = d[0].keys()

    # print out csv

    with open('GCP_pixel_indices.csv', 'w') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(d)


if __name__ == '__main__':
    main()

