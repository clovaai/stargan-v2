# AI3SW stargan-v2

## StarGANv2

See [StarGANv2 README](StarGANv2_README.md) for instructions on how to setup project and download pretrained weights for CelebA-HQ.

## Image generation using 1 source and 1 reference image

```python
source = Image.open('./assets/representative/celeba_hq/src/female/039913.jpg')
ref = Image.open('./assets/representative/celeba_hq/ref/female/015248.jpg')

args = Munch({
    'img_size': 256,
    'style_dim': 64,
    'w_hpf': 1.0,
    'latent_dim': 16,
    'num_domains': 2,
    'wing_path': './expr/checkpoints/wing.ckpt',
    'resume_iter': 100000,
    'checkpoint_dir': './expr/checkpoints/celeba_hq',
    'mode': 'sample'
})

# use the solver object as it offers some useful methods
solver = Solver(args)
solver._load_checkpoint(args.resume_iter)

# solver.predict returns generated image as a PIL image
output = solver.predict(source, ref, 'female')
```
