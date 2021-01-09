import jittor as jt
import numpy as np
from model import StyledGenerator, Discriminator
import jittor.transform as transform
from dataset import SymbolDataset
from tqdm import tqdm
import argparse
import math
import random

jt.flags.use_cuda = True
jt.flags.log_silent = True

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')
    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument('--ckpt', default=None, type=str, help='load from previous checkpoints')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    args = parser.parse_args()
    
    max_size  = 128
    init_step = int(math.log2(args.init_size) - 2)
    max_step  = int(math.log2(max_size) - 2)
    nsteps = max_step - init_step + 1

    lr = 1e-3
    mixing = True

    code_size = 512
    batch_size = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    batch_default = 32

    phase = 150_000
    max_iter = 100_000

    transform = transform.Compose([
        transform.ToPILImage(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    netG = StyledGenerator(code_dim=code_size)
    netD = Discriminator(from_rgb_activate=True)
    g_running = StyledGenerator(code_size)
    g_running.eval()

    d_optimizer = jt.optim.Adam(netD.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer = jt.optim.Adam(netG.generator.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({
        'params': netG.style.parameters(),
        'lr': lr * 0.01,
        'mult': 0.01,
        }
    )

    accumulate(g_running, netG, 0)
    
    if args.ckpt is not None:
        ckpt = jt.load(args.ckpt)

        netG.load_state_dict(ckpt['generator'])
        netD.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])

        print('resuming from checkpoint .......')
        
    ## Actual Training
    step = init_step
    resolution = int(4 * 2 ** step)
    image_loader = SymbolDataset(args.path, transform, resolution).set_attrs(
        batch_size=batch_size.get(resolution, batch_default), 
        shuffle=True
    )
    train_loader = iter(image_loader)

    requires_grad(netG, False)
    requires_grad(netD, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0
    final_progress = False
    pbar = tqdm(range(max_iter))

    for i in pbar:
        alpha = min(1, 1 / phase * (used_sample + 1))
        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1
        
        if used_sample > phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1
            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            image_loader = SymbolDataset(args.path, transform, resolution).set_attrs(
                batch_size=batch_size.get(resolution, batch_default), 
                shuffle=True
            )
            train_loader = iter(image_loader)

            jt.save(
                {
                    'generator': netG.state_dict(),
                    'discriminator': netD.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'FFHQ/checkpoint/train_step-{ckpt_step}.model',
            )

        try:
            real_image = next(train_loader)
        except (OSError, StopIteration):
            train_loader = iter(image_loader)
            real_image = next(train_loader)

        real_image.requires_grad = True
        b_size = real_image.size(0)

        real_scores = netD(real_image, step=step, alpha=alpha)
        real_predict = jt.nn.softplus(-real_scores).mean()

        grad_real = jt.grad(real_scores.sum(), real_image)
        grad_penalty = (
            grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
        ).mean()
        grad_penalty = 10 / 2 * grad_penalty

        if i % 10 == 0:
            grad_loss_val = grad_penalty.item()

        if mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(4, b_size, code_size).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        else:
            gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = netG(gen_in1, step=step, alpha=alpha)
        fake_predict = netD(fake_image, step=step, alpha=alpha)
        fake_predict = jt.nn.softplus(fake_predict).mean()

        if i % 10 == 0:
            disc_loss_val = (real_predict + fake_predict).item()

        loss_D = real_predict + grad_penalty + fake_predict
        d_optimizer.step(loss_D)

        # optimize generator

        requires_grad(netG, True)
        requires_grad(netD, False)

        fake_image = netG(gen_in2, step=step, alpha=alpha)
        predict = netD(fake_image, step=step, alpha=alpha)
        loss_G = jt.nn.softplus(-predict).mean()

        if i % 10 == 0:
            gen_loss_val = loss_G.item()

        g_optimizer.step(loss_G)

        accumulate(g_running, netG)
        requires_grad(netG, False)
        requires_grad(netD, True)

        used_sample += real_image.shape[0]

        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = (10, 5)

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.randn(gen_j, code_size), step=step, alpha=alpha
                        ).data
                    )

            jt.save_image(
                jt.concat(images, 0),
                f'FFHQ/sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 10000 == 0:
            jt.save(g_running.state_dict(), f'FFHQ/checkpoint/{str(i + 1).zfill(6)}.model')

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )
        pbar.set_description(state_msg)
