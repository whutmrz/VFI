import os
from PIL import Image
import random

def generate_black_white_image(width, height, black_ratio):
    im = Image.new('L', (width, height), color=255)
    pixels = im.load()

    for y in range(height):
        for x in range(width):
            if random.random() < black_ratio:
                pixels[x, y] = 0

    return im


if __name__ == '__main__':
    output_dir = 'black_white_images'  # 设置输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    width = 200
    height = 200
    ratios = [0.1, 0.3, 0.5, 0.6, 0.8]
    num_images_per_ratio = 20

    for ratio in ratios:
        ratio_dir = os.path.join(output_dir, f'ratio_{int(ratio * 100)}')
        if not os.path.exists(ratio_dir):
            os.makedirs(ratio_dir)

        for i in range(num_images_per_ratio):
            image = generate_black_white_image(width, height, ratio)
            filename = f'ratio_{int(ratio * 100)}_{i}.bmp'
            filepath = os.path.join(ratio_dir, filename)
            image.save(filepath)
