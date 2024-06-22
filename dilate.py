import argparse
import cv2
import os
from tqdm import tqdm
import numpy as np

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Dilate masks in a directory')
    parser.add_argument('--directory', type=str, help='Directory path containing masks')
    parser.add_argument('--kernel-size', type=int, default=5, help='Kernel size for dilation')
    parser.add_argument('--iterations', type=int, default=5, help='Number of iterations for dilation')
    return parser.parse_args()

def validate_args(directory, kernel_size, iterations):
    """Validate the provided arguments."""
    if not os.path.isdir(directory):
        raise Exception(f"Directory path '{directory}' is not valid")
    if kernel_size <= 0:
        raise Exception("Kernel size should be greater than 0")
    if iterations <= 0:
        raise Exception("Number of iterations should be greater than 0")

def dilate_images(directory, kernel_size, iterations):
    """Dilate images in the specified directory."""
    for filename in tqdm(sorted(os.listdir(directory))):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            dilated_image = cv2.dilate(image, kernel, iterations=iterations)
            cv2.imwrite(image_path, dilated_image)

def main():
    args = parse_args()
    validate_args(args.directory, args.kernel_size, args.iterations)
    dilate_images(args.directory, args.kernel_size, args.iterations)

if __name__ == '__main__':
    main()
