import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import pytorch_lightning as pl
from train_val_template import FaceCoresetNet
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
import config
import os
from utils import dotdict
import data_template as data
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from datetime import datetime
import os
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from pytorch_lightning.strategies import DDPStrategy




import sys

def debugger_is_active() -> bool:
    """Return if the debugger is currently active"""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def main(args):

    hparams = dotdict(vars(args))

    if hparams.seed is not None:
        seed_everything(hparams.seed, workers=True)

    run_name = os.path.basename(hparams.output_dir)
    wandb_mode = 'online'
    if debugger_is_active() or hparams.wandb_disable:
        wandb_mode = 'disabled'

    print('wandb_mode', wandb_mode)

    log_wandb_logger = WandbLogger(project="FaceCoresetNet", mode=wandb_mode, name=run_name, id=run_name)

    trainer_mod = FaceCoresetNet(**hparams)


    FLOP_COUNT = False
    if FLOP_COUNT:
        input = (None, None, torch.rand((1, 8000, 512)), torch.rand((1, 8000, 1)))
        trainer_mod.eval()
        flops = FlopCountAnalysis(trainer_mod, input)
        print(flop_count_table(flops))

    #log_wandb_logger.watch(trainer_mod, log='all')
    data_mod = data.DataModule(**hparams)

    save_top_k = -1
    checkpoint_callback = ModelCheckpoint(dirpath=hparams.output_dir, save_last=True,
                                          save_top_k=save_top_k)
    # create logger
    csv_logger = CSVLogger(save_dir=hparams.output_dir, name='result')
    my_loggers = [log_wandb_logger, csv_logger]
    #my_loggers = wandb_logger
    resume_from_checkpoint = hparams.resume_from_checkpoint if hparams.resume_from_checkpoint else None
    if resume_from_checkpoint is not None:
        trainer_mod = trainer_mod.load_from_checkpoint(resume_from_checkpoint,lr=hparams.lr, gamma_lr=hparams.gamma_lr,
                                                       h=hparams.h, gamma=hparams.gamma, train_data_path=hparams.train_data_path,
                                                       ijb_root=hparams.ijb_root
                                                       )

    ddp = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)

    trainer = pl.Trainer(default_root_dir=hparams.output_dir,
                         logger=my_loggers,
                         devices=hparams.devices,
                         max_epochs=hparams.epochs,
                         accelerator=hparams.accelerator,
                         strategy=ddp,
                         precision=hparams.precision,
                         fast_dev_run=hparams.fast_dev_run,
                         callbacks=[checkpoint_callback],
                         num_sanity_val_steps=0,
                         val_check_interval=hparams.val_check_interval,
                         accumulate_grad_batches=hparams.accumulate_grad_batches,
                         limit_train_batches=hparams.limit_train_batches,
                         gradient_clip_val=hparams.gradient_clip_val,
                         log_every_n_steps=10,
                         check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                         deterministic=True
                         )


    if not hparams.evaluate:
        # train / val
        print('start training')
        trainer.fit(trainer_mod, data_mod)
        print('start evaluating')
        print('evaluating from ', checkpoint_callback.best_model_path)
        #gil - remove test for now
        #trainer.test(ckpt_path='best', datamodule=data_mod)
    else:
        # eval only
        #trainer_mod.load_from_checkpoint(hparams.resume_from_checkpoint, lr=0.023, gamma_lr=0.11)
        print('start evaluating')
        trainer_mod.to('cuda:0')
        trainer_mod.on_train_epoch_end()
        #trainer.test(trainer_mod, datamodule=data_mod)


if __name__ == '__main__':

    args = config.get_args()
    # if args.wandb_disable:
    #     os.environ['WANDB_DISABLED'] = 'true'
    #wandb_disable = 'disabled' if args.wandb_disable else 'online'
    #wandb.init(project="set-face-recognition", mode='online', log_model=True,
    #                            name=run_name, id=run_name)
    #wandb.config = args
    # if args.distributed_backend == 'ddp' and args.gpus > 0:
    #     # When using a single GPU per process and per
    #     # DistributedDataParallel, we need to divide the batch size
    #     # ourselves based on the total number of GPUs we have
    #     torch.set_num_threads(1)
    #     args.total_batch_size = args.batch_size
    #     args.batch_size = int(args.batch_size / max(1, args.gpus))
    #     args.num_workers = min(args.num_workers, 16)

    if args.resume_from_checkpoint:
        assert args.resume_from_checkpoint.endswith('.ckpt')
        args.output_dir = os.path.dirname(args.resume_from_checkpoint)
        print('resume from {}'.format(args.output_dir))

    main(args)