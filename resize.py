from PIL import Image
import os

root_path = 'FFHQ/'
src_path = os.path.join(root_path, '128/')
resolutions = [64, 32, 16, 8]

for image_file in os.listdir(src_path):
    image_path = os.path.join(src_path, image_file)
    image = Image.open(image_path)
    for resolution in resolutions:
        resized_image = image.resize((resolution, resolution), Image.LANCZOS)
        target_path = root_path + str(resolution) + '/'
        resized_image.save(os.path.join(target_path, image_file))
