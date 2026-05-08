import argparse
import os
import shutil
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

os.environ["WANDB_CONSOLE"] = "off"
from pytorch_lightning.loggers import WandbLogger
from utils import instantiate_from_config


import warnings
if int(os.environ.get("LOCAL_RANK", 0)) != 0:
    warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Training Script Based on Pytorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_file = args.config
    assert os.path.exists(config_file), "The configuration file does not exist."
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    print("\n" + "*" * 10 + " Configuration " + "*" * 10)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("*" * 30 + "\n")
    experiment_name = config['log']['name']
    log_dir = os.path.join(config['log']['log_dir'], str(experiment_name))
    os.makedirs(log_dir, exist_ok=True)
    destination = os.path.join(log_dir, 'config.yaml')
    shutil.copy(config_file, destination)

    pl.seed_everything(config['seed'], workers=True)

    dm = instantiate_from_config(model_name=config['data']['target'], **config['data']['params'])
    model = instantiate_from_config(model_name=config['model']['target'], **config['model']['params'])

    ckpt_monitor = config['trainer']['ckpt_monitor']
    safe_name = ckpt_monitor.replace("/", "_")  # val_loss
    filename = f"epoch_{{epoch:03d}}_{safe_name}_{{{ckpt_monitor}:.4f}}"
    ckpt_cb = ModelCheckpoint(
        dirpath=config['trainer']['ckpt_dir'],
        filename=filename,
        auto_insert_metric_name=False,
        monitor=ckpt_monitor,
        mode=config["trainer"].get("mode", "min"),
        save_last=True,
        save_top_k=3,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    # Recommend to create a wandb account to use this
    # Or you can use a CSVLogger instead
    logger = WandbLogger(
        project=config['log']['project'],
        name=experiment_name,
        save_dir=log_dir,
        # offline=False,
        log_model=False,
        config=config
    )

    trainer = pl.Trainer(
        strategy=config['trainer'].get('strategy', 'auto'),
        devices=config['trainer']['devices'],
        accelerator='gpu',
        precision=config['trainer']['precision'],
        max_epochs=config['trainer']['max_epochs'],
        logger=logger,
        check_val_every_n_epoch=config['trainer'].get('check_val_every_n_epoch', 1),
        callbacks=[ckpt_cb, lr_cb],
        use_distributed_sampler=config.get('use_distributed_sampler', True),
        gradient_clip_val=config['trainer'].get('gradient_clip_val', None),
        gradient_clip_algorithm=config['trainer'].get('gradient_clip_algorithm', 'norm')
    )
    print("*" * 30 + "\n")

    trainer.fit(model, datamodule=dm)