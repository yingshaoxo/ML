import os
import torch
import argparse
from pytorch_lightning import (
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
)
from fengshen.data.universal_datamodule import UniversalDataModule
from fengshen.models.model_utils import (
    add_module_args,
    configure_optimizers,
    get_total_steps,
)
from fengshen.utils.universal_checkpoint import UniversalCheckpoint
from diffusers import StableDiffusionPipeline
from torch.nn import functional as F
from torchvision import transforms
from fengshen.data.taiyi_stable_diffusion_datasets.taiyi_datasets import add_data_args, load_data
import numpy as np
from PIL import Image

class Collator():
    def __init__(self, args, tokenizer):
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(
                    args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.tokenizer = tokenizer

    def __call__(self, inputs):
        max_length = min(max([len(i['caption']) for i in inputs]), 256)
        images = []
        texts = []
        for i in inputs:
            if 'npy_path' in i:
                instance_image = np.load(i['npy_path'])
            elif 'img_path' in i:
                try:
                    instance_image = Image.open(i['img_path'])
                    if not instance_image.mode == "RGB":
                        instance_image = instance_image.convert("RGB")
                except:
                    continue
            else:
                raise ValueError('no img path in samples')
            images.append(self.image_transforms(instance_image))
            texts.append(i['caption'])
        text_inputs = self.tokenizer(text=texts,
                                     images=images,
                                     max_length=max_length,
                                     padding='max_length',
                                     truncation=True,
                                     return_tensors='pt')
        # return images_input, texts_input, labels
        return {'pixel_values': torch.stack(images), 'input_ids': text_inputs['input_ids']}


class StableDiffusion(LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('Taiyi Stable Diffusion Module')
        parser.add_argument('--freeze_unet', action='store_true', default=False)
        parser.add_argument('--text_model_path',  default=None)
        parser.add_argument('--freeze_text_encoder', action='store_true', default=False)
        parser.add_argument('--use_local_token', action='store_true', default=False)
        parser.add_argument('--use_local_unet', action='store_true', default=False)
        return parent_parser

    def __init__(self, args):
        super().__init__()

        self.pipeline = StableDiffusionPipeline.from_pretrained(args.model_path)

        self.tokenizer = self.pipeline.tokenizer
        self.text_encoder = self.pipeline.text_encoder
        self.vae = self.pipeline.vae
        self.unet = self.pipeline.unet
        self.noise_scheduler = self.pipeline.scheduler

        self.pipeline.set_use_memory_efficient_attention_xformers(True)

        for param in self.vae.parameters():
            param.requires_grad = False

        if args.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        if args.freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False

        self.save_hyperparameters(args)

    def setup(self, stage) -> None:
        if stage == 'fit':
            self.total_steps = get_total_steps(self.trainer, self.hparams)
            print('Total steps: {}' .format(self.total_steps))

    def configure_optimizers(self):
        return configure_optimizers(self)

    def training_step(self, batch, batch_idx):
        latents = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * 0.18215

        # Sample noise that we'll add to the latents
        noise = torch.randn(latents.shape).to(latents.device)
        noise = noise.to(dtype=self.unet.dtype)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()
        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noisy_latents = noisy_latents.to(dtype=self.unet.dtype)

        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        # Get the text embedding for conditioning
        encoder_hidden_states = self.text_encoder(batch["input_ids"])[0]

        # Predict the noise residual
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        self.log("train_loss", loss.item())

        if self.trainer.global_rank == 0 and self.global_step == 100:
            # 打印显存占用
            from fengshen.utils.utils import report_memory
            report_memory('stable diffusion')

        return {"loss": loss}

    def on_save_checkpoint(self, checkpoint) -> None:
        if self.trainer.global_rank == 0:
            print('saving model...')
            self.pipeline.save_pretrained(os.path.join(
                args.default_root_dir, f'hf_out_{self.trainer.current_epoch}_{self.trainer.global_step}'))

    def on_load_checkpoint(self, checkpoint) -> None:
        # 兼容低版本lightning，低版本lightning从ckpt起来时steps数会被重置为0
        global_step_offset = checkpoint["global_step"]
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']
        self.trainer.fit_loop.epoch_loop._batches_that_stepped = global_step_offset


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser = add_module_args(args_parser)
    args_parser = add_data_args(args_parser)
    args_parser = UniversalDataModule.add_data_specific_args(args_parser)
    args_parser = Trainer.add_argparse_args(args_parser)
    args_parser = StableDiffusion.add_module_specific_args(args_parser)
    args_parser = UniversalCheckpoint.add_argparse_args(args_parser)
    args = args_parser.parse_args()

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = UniversalCheckpoint(args)
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=[
                                             lr_monitor,
                                             checkpoint_callback])

    model = StableDiffusion(args)
    tokenizer = model.tokenizer
    datasets = load_data(args, global_rank=trainer.global_rank)
    collate_fn = Collator(args, tokenizer)

    datamoule = UniversalDataModule(
        tokenizer=tokenizer, collate_fn=collate_fn, args=args, datasets=datasets)

    trainer.fit(model, datamoule, ckpt_path=args.load_ckpt_path)
