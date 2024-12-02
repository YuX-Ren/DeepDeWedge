import math

import pytorch_lightning as pl
import torch
import tqdm
import yaml
from .las_unet import Unet3D, EDM_Unet3D
from .fourier import apply_fourier_mask_to_tomo
from .masked_loss import masked_loss, total_variation_loss
from .missing_wedge import get_missing_wedge_mask
from .normalization import get_avg_model_input_mean_and_std_from_dataloader
from .noise_generator import noise_fbp
from .EDM_loss import EDMLoss

class ExponentialMovingAverage:
    def __init__(self, model, decay=0.999):
        """
        Args:
            model (torch.nn.Module): The model whose parameters are tracked.
            decay (float): The decay rate for the EMA. Closer to 1 means slower updates.
        """
        self.model = model
        self.decay = decay
        self.shadow_params = {name: param.clone().detach() for name, param in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        """
        Updates the EMA parameters using the model's current parameters.
        """
        for name, param in self.model.state_dict().items():
            if param.requires_grad:  # Only track parameters that require gradients
                self.shadow_params[name].mul_(self.decay).add_(param, alpha=(1.0 - self.decay))

    def apply_to(self):
        """
        Copies the EMA parameters back to the model for evaluation.
        """
        self.model.load_state_dict(self.shadow_params)

class LitUnet3D(pl.LightningModule):
    """
    PyTrochLightning 'wrapper' of a 3D U-Net. This class implements steps for model fitting, validation and logging. This class is the heart of the 'ddw fit-model' command.
    """

    def __init__(
        self,
        unet_params,
        adam_params,
        subtomo_dir,
        update_subtomo_missing_wedges_every_n_epochs=10,
        EDM = True,
    ):
        super().__init__()
        self.unet_params = unet_params
        self.adam_params = adam_params
        self.subtomo_dir = subtomo_dir
        self.update_subtomo_missing_wedges_every_n_epochs = (
            update_subtomo_missing_wedges_every_n_epochs
        )
        self.EDM = EDM
        if self.EDM:
            self.unet = EDM_Unet3D(**self.unet_params)
            self.ema = ExponentialMovingAverage(self.unet, decay=0.995)
        else:
            self.unet = Unet3D(**self.unet_params)
        self.save_hyperparameters()

    def forward(self, x):
        return self.unet(x.unsqueeze(1)).squeeze(
            1
        )  # unsqueeze to add channel dimension, squeeze to remove it

    def training_step(self, batch, batch_idx):
        if self.EDM:
            loss = self.loss_fn(self.unet, batch["model_input"])
        else:
            model_output = self(batch["model_input"]+noise_fbp(angle = batch["mw_angle"],size = batch["model_input"].shape[-1]))
            loss = masked_loss(
                model_output=model_output,
                target=batch["model_target"],
                rot_mw_mask=batch["rot_mw_mask"],
                mw_mask=batch["mw_mask"],
            )
        self.log(
            "fitting_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        # if self.EDM:
        #     pass
        # model_output = self(batch["model_input"])
        # loss = masked_loss(
        #     model_output=model_output,
        #     target=batch["model_target"],
        #     rot_mw_mask=batch["rot_mw_mask"],
        #     mw_mask=batch["mw_mask"],
        # )
        # self.log(
        #     "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        # )

    def on_before_zero_grad(self, optimizer) -> None:
        self.ema.update()

    def on_train_start(self) -> None:
        if self.current_epoch == 0:
            self.update_normalization()

    def on_train_epoch_end(self) -> None:
        pass
        # if (
        #     self.current_epoch + 1
        # ) % self.update_subtomo_missing_wedges_every_n_epochs == 0:  # +1 because the epoch indexing starts at 0
            # self.update_subtomo_missing_wedges()
            # self.update_normalization()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.adam_params)
        if self.EDM:
            def lr_lambda(current_step):
                warmup_steps = 2000  # Number of warmup steps
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            # Use LambdaLR for the scheduler
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        else:
            scheduler = None
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Step every batch
            },
        }
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        if scheduler is not None:
            scheduler.step()
    # def update_subtomo_missing_wedges(self):
    #     """
    #     Update the missing wedges of model input subtomos.
    #     """
    #     # we don't want to rotate the subtomos when updating them, so we create new dataloader objects with rotate_subtomos=False
    #     datasets = []
    #     train_loader = self.trainer.train_dataloader.loaders
    #     train_set = train_loader.dataset
    #     train_set.rotate_subtomos = False
    #     datasets.append(train_set)
    #     # val_dataloaders may be None
    #     if self.trainer.val_dataloaders is not None:
    #         val_loader = self.trainer.val_dataloaders[0]
    #         val_set = val_loader.dataset
    #         val_set.rotate_subtomos = False
    #         datasets.append(val_set)
    #     dataset = torch.utils.data.ConcatDataset(datasets)
    #     loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=train_loader.batch_size,
    #         num_workers=train_loader.num_workers,
    #     )
    #     # subtomo size has to be divisible by 2**num_downsample_layers due to U-Net architecture -> ensure this by padding
    #     subtomo_dim = dataset[0]["model_input"].shape[-1]
    #     factor = 2 ** self.unet_params["num_downsample_layers"]
    #     padding = factor * math.ceil(subtomo_dim / factor) - subtomo_dim
    #     # also make larger missing wedge mask that is compatible with the padded subtomos
    #     mw_mask = get_missing_wedge_mask(grid_size=3*[subtomo_dim + padding], mw_angle=train_set.mw_angle)
    #     with torch.no_grad():
    #         for batch in tqdm.tqdm(loader, desc="Updating subtomo missing wedges"):
    #             assert batch["rot_angle"].float().norm() == 0
    #             subtomo_batch = batch["model_input"].to(self.device)
    #             subtomo_batch = torch.nn.functional.pad(
    #                 subtomo_batch,
    #                 pad=(0, padding, 0, padding, 0, padding),
    #                 mode="constant",
    #                 value=0,
    #             )
    #             # repeat missing wedge mask for each subtomo in the batch
    #             mw_mask_batch = mw_mask.repeat((*subtomo_batch.shape[:-3], 1, 1, 1)).to(subtomo_batch.device)
    #             # forward pass
    #             subtomo_batch_ref = self.forward(subtomo_batch)
    #             # update missing wedges    
    #             subtomo_batch = apply_fourier_mask_to_tomo(
    #                 subtomo_batch, mw_mask_batch
    #             ) + apply_fourier_mask_to_tomo(subtomo_batch_ref, 1 - mw_mask_batch)
    #             # remove padding
    #             subtomo_batch = subtomo_batch[
    #                 ..., :subtomo_dim, :subtomo_dim, :subtomo_dim
    #             ]
    #             for subtomo, file in zip(subtomo_batch, batch["subtomo0_file"]):
    #                 torch.save(subtomo.cpu().clone(), file)
    #     train_set.rotate_subtomos = True
    #     if self.trainer.val_dataloaders is not None:
    #         val_set.rotate_subtomos = True

    def update_normalization(self):
        """
        Updates the average model input mean and standard deviation used to normalize the sub-tomograms.
        """
        loc, scale = get_avg_model_input_mean_and_std_from_dataloader(
            dataloader=self.trainer.train_dataloader, verbose=True
        )

        # update normalization in unet
        self.unet.normalization_loc = loc
        self.unet.normalization_scale = scale
        # update normalization in hparams
        self.unet_params["normalization_loc"] = loc
        self.unet_params["normalization_scale"] = scale
        self.update_hparam("unet_params", self.unet_params)
        self.log("normalization/loc", loc)
        self.log("normalization/scale", scale)
        if self.EDM:
            self.loss_fn = EDMLoss(data_scale=scale, data_loc=loc)

    def update_hparam(self, hparam, value):
        """
        Update a hyperparameter in the hparams.yaml file.
        """
        logger = self.trainer.logger
        logdir = f"{logger.save_dir}/{logger.name}/version_{logger.version}"
        hparams_file = f"{logdir}/hparams.yaml"
        hparams = yaml.safe_load(open(hparams_file, "r"))
        hparams[hparam] = value
        with open(hparams_file, "w") as f:
            yaml.dump(hparams, f)

