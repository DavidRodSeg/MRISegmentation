import numpy as np
import os
import cv2

parent_dir = f"{os.getcwd()}/training_data"

flair_dir = f"{parent_dir}/flair"
t1_dir = f"{parent_dir}/t1"
t1ce_dir = f"{parent_dir}/t1ce"
t2_dir = f"{parent_dir}/t2"

flair = sorted(os.listdir(flair_dir))
t1 = sorted(os.listdir(t1_dir))
t1ce = sorted(os.listdir(t1ce_dir))
t2 = sorted(os.listdir(t2_dir))

output_folder = "combined_training_data"
os.makedirs(output_folder, exist_ok=True)

for flair_img, t1_img, t1ce_img, t2_img in zip(flair, t1, t1ce, t2):
    img4 = cv2.imread(os.path.join(flair_dir, flair_img), cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread(os.path.join(t1_dir, t1_img), cv2.IMREAD_GRAYSCALE)
    img3 = cv2.imread(os.path.join(t1ce_dir, t1ce_img), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(os.path.join(t2_dir, t2_img), cv2.IMREAD_GRAYSCALE)

    combined_img = cv2.merge([img1, img2, img3, img4])
    
    output_path = os.path.join(output_folder, t1_img)
    cv2.imwrite(output_path, combined_img)