from PIL import Image
import argparse
import os
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce image size to exponential of 2 until min_size (default to 8)')
    parser.add_argument('--min_size', type=int, default=8, help='minimal size of the result image')
    parser.add_argument('path', type=str, help='path to image files')
    
    args = parser.parse_args()
    
    root_path = os.path.split(os.path.abspath(args.path))[0]
    image_file = os.listdir(args.path)[0]
    image_path = os.path.join(args.path, image_file)
    image = Image.open(image_path)
    max_size = image.size[0]
    size = args.min_size
    
    while size < max_size:        
        os.mkdir(root_path + '/' + str(size))
        size = size * 2
    
    for image_file in os.listdir(args.path):
        image_path = os.path.join(args.path, image_file)
        
        ext = os.path.splitext(image_path)[-1]
        if ext not in ['.png', '.jpg']:
            continue
            
        image = Image.open(image_path)
        size = args.min_size
        
        while size < max_size:
            resized_image = image.resize((size, size), Image.LANCZOS)
            target_path = os.path.join(root_path, str(size), image_file)
            resized_image.save(target_path)
            size = size * 2
            