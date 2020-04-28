"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import numpy as np
import torch

from metrics.fid import calculate_fid_given_paths
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils


@torch.no_grad()
def calculate_metrics(nets, args, step, mode):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    domains = os.listdir(args.val_img_dir)
    domains.sort()
    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    lpips_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]

        if mode == 'reference':
            path_ref = os.path.join(args.val_img_dir, trg_domain)
            loader_ref = get_eval_loader(root=path_ref,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False,
                                         drop_last=True)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         imagenet_normalize=False)

            task = '%s2%s' % (src_domain, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            lpips_values = []
            print('Generating images and calculating LPIPS for %s...' % task)
            for i, x_src in enumerate(tqdm(loader_src, total=len(loader_src))):
                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                # generate 10 outputs from the same input
                group_of_images = []
                for j in range(args.num_outs_per_domain):
                    if mode == 'latent':
                        z_trg = torch.randn(N, args.latent_dim).to(device)
                        s_trg = nets.mapping_network(z_trg, y_trg)
                    else:
                        try:
                            x_ref = next(iter_ref).to(device)
                        except:
                            iter_ref = iter(loader_ref)
                            x_ref = next(iter_ref).to(device)

                        if x_ref.size(0) > N:
                            x_ref = x_ref[:N]
                        s_trg = nets.style_encoder(x_ref, y_trg)

                    x_fake = nets.generator(x_src, s_trg, masks=masks)
                    group_of_images.append(x_fake)

                    # save generated images to calculate FID later
                    for k in range(N):
                        filename = os.path.join(
                            path_fake,
                            '%.4i_%.2i.png' % (i*args.val_batch_size+(k+1), j+1))
                        utils.save_image(x_fake[k], ncol=1, filename=filename)

                lpips_value = calculate_lpips_given_images(group_of_images)
                lpips_values.append(lpips_value)

            # calculate LPIPS for each task (e.g. cat2dog, dog2cat)
            lpips_mean = np.array(lpips_values).mean()
            lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean

        # delete dataloaders
        del loader_src
        if mode == 'reference':
            del loader_ref
            del iter_ref

    # calculate the average LPIPS for all tasks
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    utils.save_json(lpips_dict, filename)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode)


def calculate_fid_for_all_tasks(args, domains, step, mode):
    print('Calculating FID for all tasks...')
    fid_values = OrderedDict()
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            task = '%s2%s' % (src_domain, trg_domain)
            path_real = os.path.join(args.train_img_dir, trg_domain)
            path_fake = os.path.join(args.eval_dir, task)
            print('Calculating FID for %s...' % task)
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                img_size=args.img_size,
                batch_size=args.val_batch_size)
            fid_values['FID_%s/%s' % (mode, task)] = fid_value

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value / len(fid_values)
    fid_values['FID_%s/mean' % mode] = fid_mean

    # report FID values
    filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.json' % (step, mode))
    utils.save_json(fid_values, filename)
