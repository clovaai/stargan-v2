import os
import argparse
import tempfile
from pathlib import Path
import shutil
from munch import Munch
from torch.backends import cudnn
import torch
import cog
from core.data_loader import InputFetcher
from core.data_loader import get_test_loader
from core.solver import Solver
from core.utils import save_image


class Predictor(cog.Predictor):
    def setup(self):
        self.args = parse_args()

    @cog.input(
        "model",
        type=str,
        options=["celeb", "animal"],
        default="animal",
        help="use celeb (celeba_hq) or animal (afhq) model",
    )
    @cog.input(
        "src",
        type=Path,
        help="source facial image (decides pose for animals or identity for humans)",
    )
    @cog.input(
        "ref",
        type=Path,
        help="reference facial image (decides breed/style for animals or gender/style for humans)",
    )
    def predict(self, src, ref, model):
        try:
            src_dir = "input/cog_temp/src/dummy"
            ref_dir = "input/cog_temp/ref/dummy"
            os.makedirs(src_dir, exist_ok=True)
            os.makedirs(ref_dir, exist_ok=True)
            self.args.w_hpf = 0 if model == "animal" else 1
            self.args.checkpoint_dir = (
                "expr/checkpoints/afhq"
                if model == "animal"
                else "expr/checkpoints/celeba_hq"
            )
            src_path = os.path.join(src_dir, os.path.basename(src))
            ref_path = os.path.join(ref_dir, os.path.basename(ref))
            shutil.copy(str(src), src_path)
            shutil.copy(str(ref), ref_path)

            self.args.src_dir = "input/cog_temp/src/"
            self.args.ref_dir = "input/cog_temp/ref/"
            # args.num_domains = 1
            out_path = Path(tempfile.mkdtemp()) / "out.png"

            cudnn.benchmark = True
            torch.manual_seed(self.args.seed)

            solver = SingleSolver(self.args)
            # we only use sample mode

            loaders = Munch(
                src=get_test_loader(
                    root=self.args.src_dir,
                    img_size=self.args.img_size,
                    batch_size=self.args.val_batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                ),
                ref=get_test_loader(
                    root=self.args.ref_dir,
                    img_size=self.args.img_size,
                    batch_size=self.args.val_batch_size,
                    shuffle=False,
                    num_workers=self.args.num_workers,
                ),
            )
            solver.single_sample(loaders, str(out_path))
        finally:
            clean_folder(src_dir)
            clean_folder(ref_dir)
        return out_path


def parse_args():
    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument("--img_size", type=int, default=256, help="Image resolution")
    parser.add_argument("--num_domains", type=int, default=1, help="Number of domains")
    parser.add_argument(
        "--latent_dim", type=int, default=16, help="Latent vector dimension"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension of mapping network",
    )
    parser.add_argument(
        "--style_dim", type=int, default=64, help="Style code dimension"
    )

    # weight for objective functions
    parser.add_argument(
        "--lambda_reg", type=float, default=1, help="Weight for R1 regularization"
    )
    parser.add_argument(
        "--lambda_cyc", type=float, default=1, help="Weight for cyclic consistency loss"
    )
    parser.add_argument(
        "--lambda_sty",
        type=float,
        default=1,
        help="Weight for style reconstruction loss",
    )
    parser.add_argument(
        "--lambda_ds", type=float, default=1, help="Weight for diversity sensitive loss"
    )
    parser.add_argument(
        "--ds_iter",
        type=int,
        default=100000,
        help="Number of iterations to optimize diversity sensitive loss",
    )
    parser.add_argument(
        "--w_hpf", type=float, default=1, help="weight for high-pass filtering"
    )

    # training arguments
    parser.add_argument(
        "--randcrop_prob",
        type=float,
        default=0.5,
        help="Probabilty of using random-resized cropping",
    )
    parser.add_argument(
        "--total_iters", type=int, default=100000, help="Number of total iterations"
    )
    parser.add_argument(
        "--resume_iter",
        type=int,
        default=100000,
        help="Iterations to resume training/testing",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--val_batch_size", type=int, default=32, help="Batch size for validation"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for D, E and G"
    )
    parser.add_argument("--f_lr", type=float, default=1e-6, help="Learning rate for F")
    parser.add_argument(
        "--beta1", type=float, default=0.0, help="Decay rate for 1st moment of Adam"
    )
    parser.add_argument(
        "--beta2", type=float, default=0.99, help="Decay rate for 2nd moment of Adam"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--num_outs_per_domain",
        type=int,
        default=10,
        help="Number of generated images per domain during sampling",
    )

    # misc
    parser.add_argument(
        "--mode",
        type=str,
        default="sample",
        choices=["train", "sample", "eval", "align"],
        help="This argument is used in solver",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers used in DataLoader",
    )
    parser.add_argument(
        "--seed", type=int, default=777, help="Seed for random number generator"
    )

    # directory for training
    parser.add_argument(
        "--train_img_dir",
        type=str,
        default="data/celeba_hq/train",
        help="Directory containing training images",
    )
    parser.add_argument(
        "--val_img_dir",
        type=str,
        default="data/celeba_hq/val",
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default="expr/samples",
        help="Directory for saving generated images",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="expr/checkpoints",
        help="Directory for saving network checkpoints",
    )

    # directory for calculating metrics
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="expr/eval",
        help="Directory for saving metrics, i.e., FID and LPIPS",
    )

    # directory for testing
    parser.add_argument(
        "--result_dir",
        type=str,
        default="expr/results",
        help="Directory for saving generated images and videos",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="assets/representative/celeba_hq/src",
        help="Directory containing input source images",
    )
    parser.add_argument(
        "--ref_dir",
        type=str,
        default="assets/representative/celeba_hq/ref",
        help="Directory containing input reference images",
    )
    parser.add_argument(
        "--inp_dir",
        type=str,
        default="assets/representative/custom/female",
        help="input directory when aligning faces",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="assets/representative/celeba_hq/src/female",
        help="output directory when aligning faces",
    )

    # face alignment
    parser.add_argument("--wing_path", type=str, default="expr/checkpoints/wing.ckpt")
    parser.add_argument(
        "--lm_path", type=str, default="expr/checkpoints/celeba_lm_mean.npz"
    )

    # step size
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--eval_every", type=int, default=50000)

    args = parser.parse_args("")
    return args


class SingleSolver(Solver):
    @torch.no_grad()
    def single_sample(self, loaders, out_path):
        args = self.args
        nets_ema = self.nets_ema
        self._load_checkpoint(args.resume_iter)

        src = next(InputFetcher(loaders.src, None, args.latent_dim, "test"))
        ref = next(InputFetcher(loaders.ref, None, args.latent_dim, "test"))

        print("Working on {}...".format(out_path))
        save_translated_image(nets_ema, args, src.x, ref.x, ref.y, out_path)


def save_translated_image(nets, args, x_src, x_ref, y_ref, filename):
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    s_ref = nets.style_encoder(x_ref, y_ref)
    x_fake = nets.generator(x_src, s_ref, masks=masks)
    save_image(x_fake, 1, filename)


def clean_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
