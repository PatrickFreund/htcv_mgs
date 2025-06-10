"""
This module defines image preprocessing and augmentation routines.

Functions:
- get_train_transforms(...) : returns a transform pipeline with augmentation
- get_val_transforms(...)   : returns a transform pipeline without augmentation
- compute_dataset_stats(...) : computes mean/std for normalization
- load_dataset_stats(...)    : loads stats from a JSON file
"""

import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
import math
import json

# ------------------------------------------------------------------
# Augmentation Utility Functions
# ------------------------------------------------------------------

def crop_to_max_square_and_resize(image):
    """
    Crop the largest possible square from the center of the image,
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    width, height = image.size
    min_side = min(width, height)

    center_x, center_y = width // 2, height // 2

    left = center_x - min_side // 2
    top = center_y - min_side // 2
    right = left + min_side
    bottom = top + min_side

    return image.crop((left, top, right, bottom))

def get_max_inscribed_rect(width, height, angle_degrees):
    """
    Calculate the maximum inscribed rectangle for a rotated image.
    """
    # Convert angle to radians and ensure it's in the first quadrant
    angle = math.radians(angle_degrees)
    angle = abs(angle) % (math.pi/2)
    
    # Determine which dimension is longer
    width_is_longer = width >= height
    side_long, side_short = (width, height) if width_is_longer else (height, width)
    
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    
    if side_short <= 2 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        x = 0.5 * side_short
        wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
    else:
        cos_2a = cos_a * cos_a - sin_a * sin_a
        margin = 0.98
        wr = (width * cos_a - height * sin_a) / cos_2a * margin
        hr = (height * cos_a - width * sin_a) / cos_2a * margin
    
    return wr, hr

def crop_to_square(image, wr, hr):
    """
    Crop an image to the maximum inscribed rectangle and then to a square.
    """
    # Convert numpy array to PIL image if it's not already
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Get image dimensions
    width, height = image.size
    
    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2
    
    # First crop to the maximum inscribed rectangle
    # Calculate the coordinates for the rectangle crop
    rect_x1 = max(center_x - int(wr // 2), 0)
    rect_y1 = max(center_y - int(hr // 2), 0)
    rect_x2 = min(rect_x1 + int(wr), width)
    rect_y2 = min(rect_y1 + int(hr), height)
    
    # Crop to the maximum inscribed rectangle
    rect_image = image.crop((rect_x1, rect_y1, rect_x2, rect_y2))
    
    # Now crop to square if needed
    rect_width, rect_height = rect_image.size
    if rect_width == rect_height:
        return rect_image
    
    # Calculate the square crop
    min_side = min(rect_width, rect_height)
    sq_center_x, sq_center_y = rect_width // 2, rect_height // 2
    
    sq_x1 = max(sq_center_x - min_side // 2, 0)
    sq_y1 = max(sq_center_y - min_side // 2, 0)
    sq_x2 = min(sq_x1 + min_side, rect_width)
    sq_y2 = min(sq_y1 + min_side, rect_height)
    
    # Crop to square
    return rect_image.crop((sq_x1, sq_y1, sq_x2, sq_y2))

def apply_rotation_crop(image, angle_range=(0, 7)):
    """
    Apply random rotation to an image and crop out black edges.
    """
    # Apply rotation with random angle
    angle_degrees = random.uniform(angle_range[0], angle_range[1])
    rotated_image = transforms.functional.rotate(
        image, 
        angle_degrees,
        fill=0 # fills the background black
    )

    # Calculate the maximum inscribed rectangle and then
    # crop to square so no black edges are left
    wr, hr = get_max_inscribed_rect(
        rotated_image.size[0],
        rotated_image.size[1],
        angle_degrees
    )
    square_image = crop_to_square(rotated_image, wr, hr)
    
    return square_image

class RotateCrop:
    def __init__(self, angle_range=(0, 7), flip_prob=0.5):
        self.angle_range = angle_range
        self.flip_prob = flip_prob

    def __call__(self, image):
        angle_degrees = random.uniform(self.angle_range[0], self.angle_range[1])
        if random.random() < self.flip_prob:
            image = transforms.functional.hflip(image)
        image = apply_rotation_crop(image, angle_range=(angle_degrees, angle_degrees))
        return image

class CenterCrop:
    def __call__(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return crop_to_max_square_and_resize(image)


# ------------------------------------------------------------------
# Transform Functions
# ------------------------------------------------------------------

def get_train_transforms(
    output_size=(224, 224), 
    flip_prob=0.5, 
    angle_range=(0, 7), 
    mean=0.5, 
    std=0.5
):
    return transforms.Compose([
        RotateCrop(angle_range=angle_range, flip_prob=flip_prob),
        transforms.Resize(output_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

def get_val_transforms(output_size=(224, 224), mean=0.5, std=0.5):
    """
    Get a custom transform function for validation. Preprossesing only.
    """
    return transforms.Compose([
        CenterCrop(),
        transforms.Resize(output_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])

# ------------------------------------------------------------------
# Dataset Statistics
# ------------------------------------------------------------------

def compute_dataset_stats(image_folder):
    """
    Compute mean and standard deviation of a dataset of grayscale images.
    """
    # Initialize variables to accumulate pixel values
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    num_pixels = 0
    
    # Process each image in the folder
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')):
            img_path = os.path.join(image_folder, filename)
            try:
                # Open image and convert to numpy array
                image = Image.open(img_path)
                image_array = np.array(image).astype(np.float32) / 255.0  # Normalize to [0,1]
                
                # Accumulate statistics
                pixel_sum += np.sum(image_array)
                pixel_squared_sum += np.sum(image_array ** 2)
                num_pixels += image_array.size
                
            except Exception as e:
                print(f"Failed to process image {filename} for stats: {e}. Skipping.")
                continue
    
    # Calculate mean and standard deviation
    mean = pixel_sum / num_pixels
    var = (pixel_squared_sum / num_pixels) - (mean ** 2)
    std = np.sqrt(var)
    
    print(f"Dataset statistics - Mean: {mean:.4f}, Std: {std:.4f}")
    save_dataset_stats(mean, std, 'data/dataset_stats.json')
    return mean, std

def save_dataset_stats(mean, std, output_file='dataset_stats.json'):
    """
    Save dataset statistics to a JSON file for later use.
    """
    stats = {'mean': float(mean), 'std': float(std)}
    with open(output_file, 'w') as f:
        json.dump(stats, f)
    print(f"Saved dataset statistics to {output_file}")

def load_dataset_stats(stats_path='data/dataset_stats.json'):
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['mean'], stats['std']

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", 
        type=str, 
        required=True, 
        help="Path to directory with images"
    )
    args = parser.parse_args()

    compute_dataset_stats(args.image_dir)