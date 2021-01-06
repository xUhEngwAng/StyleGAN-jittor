import argparse
import math

import jittor as jt
jt.flags.use_cuda = True
jt.flags.log_silent = True

from model import StyledGenerator

@jt.no_grad()
def get_mean_style(generator):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(jt.randn(1024, 512))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@jt.no_grad()
def sample(generator, step, mean_style, n_sample):
    image = generator(
        jt.randn(n_sample, 512),
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=0.7,
    )
    
    return image

@jt.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target):
    source_code = jt.randn(n_source, 512)
    target_code = jt.randn(n_target, 512)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [jt.ones((1, 3, shape, shape)) * -1]

    source_image = generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )
    target_image = generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    )

    images.append(source_image)

    for i in range(n_target):
        image = generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=0.7,
            mixing_range=(0, 1),
        )
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = jt.concat(images, 0)
    
    return images


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=128, help='size of the image')
    parser.add_argument('--n_row', type=int, default=5, help='number of rows of sample matrix')
    parser.add_argument('--n_col', type=int, default=5, help='number of columns of sample matrix')
    parser.add_argument('path', type=str, help='path to checkpoint file')
    
    args = parser.parse_args()

    generator = StyledGenerator(512)
    ckpt = jt.load(args.path)
    generator.load_state_dict(ckpt)
    generator.eval()

    mean_style = get_mean_style(generator)

    step = int(math.log(args.size, 2)) - 2
    
    img = sample(generator, step, mean_style, args.n_row * args.n_col)
    jt.save_image(img, 'FFHQ/style_mixing/sample.png', nrow=args.n_col, normalize=True, range=(-1, 1))
    
    for j in range(20):
        img = style_mixing(generator, step, mean_style, args.n_col, args.n_row)
        jt.save_image(
            img, f'FFHQ/style_mixing/sample_mixing_{j}.png', nrow=args.n_col + 1, normalize=True, range=(-1, 1)
        )