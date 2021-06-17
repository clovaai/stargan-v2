# AI3SW stargan-v2

## StarGANv2

See [StarGANv2 README](StarGANv2_README.md) for instructions on how to setup project and download pretrained weights for CelebA-HQ.

## Additional Details

* Due to an error where the latest commit in [StarGANv2's](https://github.com/clovaai/stargan-v2) master branch does not work with the download pre-trained network for CelebA-HQ, this project uses an earlier commit in the master branch: `e28bdee908fdf3cb787cc2dfa3310d586e4b60ed`
* An issue has already been logged on the official repository: [Error when loading state_dict for Generator #103](https://github.com/clovaai/stargan-v2/issues/103)

## Python Environment Management using Conda

```bash
$ # create conda environment
$ conda env create --file environment.yml
$ conda activate stargan-v2
```

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

* Notebook demo can be found [here](notebooks/Predict.ipynb) in the `Proper Usage` section.

## Flask App

A Flask application can be setup that creates a `predict` endpoint that takes in:

* a source image encoded in base64 format
* a reference image encoded in base64 format
* class of reference image: `male` or `female`

and returns a generated image encoded in base64 format.

For a demo of the `predict` endpoint, refer to [Test_flask_app](notebooks/Test_flask_app.ipynb) notebook.

### Before Running

1. Download pre-trained networks for StarGANv2:

    ```bash
    bash download.sh pretrained-network-celeba-hq
    bash download.sh wing
    ```

1. Update `MODEL_ARGS` in [`config.py`](config.py) with path to pre-trained networks:

    * `wing_path`
    * `checkpoint_dir`

1. Create `instance` directory in project folder

1. Create ["deployment specific"](https://flask.palletsprojects.com/en/2.0.x/config/#instance-folders) config file `config.py` in `instance` directory with the following variables:
    * `SECRET_KEY`

### Using Flask Built-in Development Server

```bash
$ export FLASK_APP=flask_app
$ # export FLASK_ENV=development
$ flask run --host=0.0.0.0 --port=5000
```