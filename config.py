import os
import logging

def instance_id_path():
    ID = '0x1' # default value
    for file in os.listdir(os.getcwd()):
        if file.endswith(".log") and not file == "log.log":  # ignore base .log file
            try:
                ID = hex(int(file[:-4],16) + 1) # remove the extension, increment by 1, convert to hex
                os.remove(file)  # if exception not raised, then delete file
            except:
                continue  # check the next file

    ID_path = './' + ID + '.log'

    return ID_path

logging.addLevelName(9,"AddINFO")  # Create a new level below DEBUG solely for additional info

def AddINFO(self, message, *args, **kws):
    if self.isEnabledFor(9):
        self._log(9, message, args, **kws)

logging.Logger.AddINFO = AddINFO
# Create message through: logging.getLogger().AddINFO('YOUR MESSAGE')

LOGGING_CONFIG = {
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s.%(msecs)03d] %(levelname)s in %(module)s: %(message)s',
        'datefmt': '%d/%m/%Y %H:%M:%S'
    }},
    'handlers': {'console': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://sys.stdout',
        'formatter': 'default',
        'level': logging.INFO

    }, 'file': {
        'class': 'logging.FileHandler',
        'formatter': 'default',
        'filename': './log.log',
        'level': logging.DEBUG
    },
        'instance': {
        'class': 'logging.FileHandler',
        'formatter': 'default',
        'filename': instance_id_path(),
        'level': 9  # Level == Additional Info; only 'instance' handler will be able to parse AddINFO messages
    }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file','instance']
    }
}

MODEL_ARGS = {
    'img_size': 256,
    'style_dim': 64,
    'w_hpf': 1.0,
    'latent_dim': 16,
    'num_domains': 2,
    'wing_path': './expr/checkpoints/wing.ckpt',
    'lm_path': './expr/checkpoints/celeba_lm_mean.npz',
    'resume_iter': 100000,
    'checkpoint_dir': './expr/checkpoints/celeba_hq',
    'mode': 'sample'
}
