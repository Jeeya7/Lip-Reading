import random
import cv2 as cv
import csv 
import os

# Load test images
def get_color_image_paths(csv_path):
    image_paths = []
    with open (csv_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        folder_paths = [row[0] for row in csv_reader]
    for folder in folder_paths:
        if os.path.isdir(folder):
            for image in os.listdir(folder):
                if "color" in image:
                    full_path = os.path.join(folder, image)
                    image_paths.append(full_path)
                    
    return image_paths
    
def random_color_images_batch(batch_size, csv_path):
    image_paths = get_color_image_paths(csv_path)
    selected_paths = random.sample(image_paths, min(batch_size, len(image_paths)))
    images = []
    for path in selected_paths:
        img = cv.imread(path)
        if img is not None:
            images.append(img)
    return images
        
def random_image(csv_path):
    images = random_color_images_batch(batch_size=1, csv_path=csv_path)
    return images[0] if images else None
