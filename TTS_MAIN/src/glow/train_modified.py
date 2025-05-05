import os
import json
import argparse
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp

from data_utils_temp import TextMelLoader, TextMelCollate
import models_modified
import common_modified
import utils


print(f"Using models_modified from: {models_modified.__file__}")

global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()-5
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "80000"

    hps = utils.get_hparams()
    mp.spawn(
        train_and_eval,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hps,
        ),
    )


def train_and_eval(rank, n_gpus, hps):
    global global_step
    logger = None
    if rank == 0:
        logger = utils.get_logger(hps.log_dir)
        logger.info(hps)
        utils.check_git_hash(hps.log_dir)
        writer = SummaryWriter(log_dir=hps.log_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.log_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)


    train_dataset = TextMelLoader(hps.data.training_files, hps.data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=n_gpus, rank=rank, shuffle=True
    )
    collate_fn = TextMelCollate(1) # Instantiate the modified collate function
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        batch_size=hps.train.batch_size,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        sampler=train_sampler,
    )
    if rank == 0:
        val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
        val_loader = DataLoader(
            val_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
    symbols = hps.data.punc + hps.data.chars
    generator = models_modified.FlowGenerator(
        n_vocab=len(symbols) + getattr(hps.data, "add_blank", False),
        out_channels=hps.data.n_mel_channels,
        **hps.model
    ).cuda(rank)
    optimizer_g = common_modified.Adam(
        generator.parameters(),
        scheduler=hps.train.scheduler,
        dim_model=hps.model.hidden_channels,
        warmup_steps=hps.train.warmup_steps,
        lr=hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    if hps.train.fp16_run:
        generator, optimizer_g._optim = amp.initialize(
            generator, optimizer_g._optim, opt_level="O1"
        )
    generator = DDP(generator)
    epoch_str = 1
    global_step = 0
    try:

        checkpoint_path = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        if checkpoint_path:
            _, _, _, epoch_str = utils.load_checkpoint(
                checkpoint_path, generator, optimizer_g
            )
            epoch_str += 1
            optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
            optimizer_g._update_learning_rate()
            global_step = (epoch_str - 1) * len(train_loader)
        else:
            # If no checkpoint, start from scratch or use ddi_G.pth if it exists
            if hps.train.ddi and os.path.isfile(os.path.join(hps.model_dir, "ddi_G.pth")):
                _ = utils.load_checkpoint(
                    os.path.join(hps.model_dir, "ddi_G.pth"), generator, optimizer_g
                )
    except Exception as e:
        logger.error(f"Error while loading checkpoint: {e}")
        # Proceed with training from scratch if no checkpoint exists

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank == 0:
            train(
                rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer
            )
            evaluate(
                rank,
                epoch,
                hps,
                generator,
                optimizer_g,
                val_loader,
                logger,
                writer_eval,
            )
            if epoch % hps.train.save_epoch == 0:
                utils.save_checkpoint(
                    generator,
                    optimizer_g,
                    hps.train.learning_rate,
                    epoch,
                    os.path.join(hps.model_dir, "G_{}.pth".format(epoch)),
                )
        else:
            train(rank, epoch, hps, generator, optimizer_g, train_loader, None, None)

def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
    train_loader.sampler.set_epoch(epoch)
    global global_step

    generator.train()
    for batch_idx, (x, x_lengths, y, y_lengths, pitch, energy) in enumerate(train_loader):
        # Move tensors to GPU
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

        # Move pitch and energy to GPU
        pitch = pitch.cuda(rank, non_blocking=True)
        energy = energy.cuda(rank, non_blocking=True)

        # Train Generator
        optimizer_g.zero_grad()

        (
            (z, z_m, z_logs, logdet, z_mask),
            (x_m, x_logs, x_mask),
            (attn, logw, logw_),
            pred_pitch,
            pred_energy,
        ) = generator(x, x_lengths, y, y_lengths, pitch=pitch, energy=energy, gen=False)

        # Calculate losses
        l_mle = common_modified.mle_loss(z, z_m, z_logs, logdet, z_mask)
        l_length = common_modified.duration_loss(logw, logw_, x_lengths)

        # Generate mask
        # y_max_length = y.size(2)
        # ... inside the train function ...

        y_max_length = y.size(2)
        y_mask = common_modified.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(pitch.dtype)

        l_pitch = common_modified.pitch_loss(pred_pitch, pitch, y_mask.squeeze(1))
        l_energy = common_modified.energy_loss(pred_energy, energy, y_mask.squeeze(1))


        # y_max_length = y.size(2)
        # y_mask = common_modified.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(pitch.dtype)

        # # Align the target pitch to the decoder length using the attention weights
        # with torch.no_grad():
        #     attn_squeeze = attn.squeeze(1)  # [b, t_dec, t_enc]
        #     pitch_squeezed = pitch.squeeze(1) # [b, t_enc]
        #     pitch_unsqueezed = pitch_squeezed.unsqueeze(-1) # [b, t_enc, 1]

        #     print(f"Shape of attn_squeeze in train: {attn_squeeze.shape}")
        #     print(f"Shape of pitch_unsqueezed in train: {pitch_unsqueezed.shape}")

        #     aligned_pitch = torch.matmul(attn_squeeze, pitch_unsqueezed).unsqueeze(1) # [b, t_dec, 1] -> [b, 1, t_dec]

        # l_pitch = common_modified.pitch_loss(outputs[3], aligned_pitch, y_mask.squeeze(1))
        # l_energy = common_modified.energy_loss(outputs[4], energy, y_mask.squeeze(1))

        # ... rest of your training code ...

        # y_mask = common_modified.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(pitch.dtype)

        # # *** INSERT PRINT STATEMENTS HERE ***
        # print(f"Shape of pred_pitch: {pred_pitch.shape}")
        # print(f"Shape of pitch: {pitch.shape}")
        # print(f"Shape of y_mask: {y_mask.squeeze(1).shape}")

        # l_pitch = common_modified.pitch_loss(pred_pitch, pitch, y_mask.squeeze(1))
        # l_energy = common_modified.energy_loss(pred_energy, energy, y_mask.squeeze(1))

        loss_gs = [l_mle, l_length, l_pitch, l_energy]
        loss_g = sum(loss_gs)

# def train(rank, epoch, hps, generator, optimizer_g, train_loader, logger, writer):
#     train_loader.sampler.set_epoch(epoch)
#     global global_step

#     generator.train()
#     for batch_idx, (x, x_lengths, y, y_lengths, pitch, energy) in enumerate(train_loader):
#         # Move tensors to GPU
#         x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
#         y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        
#         # Move pitch and energy to GPU
#         pitch = pitch.cuda(rank, non_blocking=True)
#         energy = energy.cuda(rank, non_blocking=True)

#         # Train Generator
#         optimizer_g.zero_grad()

#         (
#             (z, z_m, z_logs, logdet, z_mask),
#             (x_m, x_logs, x_mask),
#             (attn, logw, logw_),
#             pred_pitch,
#             pred_energy,
#         ) = generator(x, x_lengths, y, y_lengths, pitch=pitch, energy=energy, gen=False)

#         # Calculate losses
#         l_mle = common_modified.mle_loss(z, z_m, z_logs, logdet, z_mask)
#         l_length = common_modified.duration_loss(logw, logw_, x_lengths)

#         # ... inside the train function ...

#         y_max_length = y.size(2)
#         y_mask = common_modified.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(pitch.dtype)

#         l_pitch = common_modified.pitch_loss(pred_pitch, pitch, y_mask.squeeze(1))
#         l_energy = common_modified.energy_loss(pred_energy, energy, y_mask.squeeze(1))

#     # ... rest of your training code ...

#         # l_pitch = common_modified.pitch_loss(pred_pitch, pitch)
#         # l_energy = common_modified.energy_loss(pred_energy, energy)

#         loss_gs = [l_mle, l_length, l_pitch, l_energy]
#         loss_g = sum(loss_gs)

        # FP16 training
        if hps.train.fp16_run:
            with amp.scale_loss(loss_g, optimizer_g._optim) as scaled_loss:
                scaled_loss.backward()
            grad_norm = common_modified.clip_grad_value_(
                amp.master_params(optimizer_g._optim), 5
            )
        else:
            loss_g.backward()
            grad_norm = common_modified.clip_grad_value_(generator.parameters(), 5)
        optimizer_g.step()

        if rank == 0:
            # Logging and visualization
            if batch_idx % hps.train.log_interval == 0:
                # If using DDP, use module attribute to access the generator
                if isinstance(generator, torch.nn.parallel.DistributedDataParallel):
                    y_gen, *_ = generator.module(x[:1], x_lengths[:1], gen=True)
                else:
                    y_gen, *_ = generator(x[:1], x_lengths[:1], gen=True)

                # Logging train progress
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(x),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss_g.item(),
                    )
                )
                logger.info(
                    [x.item() for x in loss_gs] + [global_step, optimizer_g.param_groups[0]["lr"]]
                )

                # Update scalar dictionary with losses
                scalar_dict = {
                    "loss/g/total": loss_g,
                    "learning_rate": optimizer_g.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                }

                scalar_dict.update({
                    "loss/g/pitch": l_pitch,
                    "loss/g/energy": l_energy
                })

                # Summarize results
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images={
                        "y_org": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
                        "y_gen": utils.plot_spectrogram_to_numpy(y_gen[0].data.cpu().numpy()),
                        "attn": utils.plot_alignment_to_numpy(attn[0, 0].data.cpu().numpy()),
                    },
                    scalars=scalar_dict,
                )

        # Increment global step after every batch
        global_step += 1

    # Epoch summary logging
    if rank == 0:
        logger.info("====> Epoch: {}".format(epoch))


def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
    if rank == 0:
        global global_step
        generator.eval()
        losses_tot = []

        with torch.no_grad():
            for batch_idx, (x, x_lengths, y, y_lengths, pitch, energy) in enumerate(val_loader):
                # Move tensors to GPU
                x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
                y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

                # Move pitch and energy to GPU
                pitch = pitch.cuda(rank, non_blocking=True)
                energy = energy.cuda(rank, non_blocking=True)

                # Evaluate Generator
                (
                    (z, z_m, z_logs, logdet, z_mask),
                    (x_m, x_logs, x_mask),
                    (attn, logw, logw_),
                    pred_pitch,
                    pred_energy,
                ) = generator(x, x_lengths, y, y_lengths, pitch=pitch, energy=energy, gen=False)

                # Calculate losses
                y_max_length = y.size(2)
                y_mask = common_modified.sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(pitch.dtype)

                l_mle = common_modified.mle_loss(z, z_m, z_logs, logdet, z_mask)
                l_length = common_modified.duration_loss(logw, logw_, x_lengths)
                l_pitch = common_modified.pitch_loss(pred_pitch, pitch, y_mask.squeeze(1))
                l_energy = common_modified.energy_loss(pred_energy, energy, y_mask.squeeze(1))

                loss_gs = [l_mle, l_length, l_pitch, l_energy]
                loss_g = sum(loss_gs)

                # Accumulate losses over the batches
                if batch_idx == 0:
                    losses_tot = loss_gs
                else:
                    losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

                # Log progress
                if batch_idx % hps.train.log_interval == 0:
                    logger.info(
                        "Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(x),
                            len(val_loader.dataset),
                            100.0 * batch_idx / len(val_loader),
                            loss_g.item(),
                        )
                    )
                    logger.info([x.item() for x in loss_gs])

        # Normalize and log total loss
        losses_tot = [x / len(val_loader) for x in losses_tot]
        loss_tot = sum(losses_tot)

        scalar_dict = {"loss/g/total": loss_tot}
        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})

        # Write evaluation results
        utils.summarize(
            writer=writer_eval, global_step=global_step, scalars=scalar_dict
        )

        # Log epoch summary
        logger.info("====> Epoch: {}".format(epoch))


# def evaluate(rank, epoch, hps, generator, optimizer_g, val_loader, logger, writer_eval):
#     if rank == 0:
#         global global_step
#         generator.eval()
#         losses_tot = []

#         with torch.no_grad():
#             for batch_idx, (x, x_lengths, y, y_lengths, pitch, energy) in enumerate(val_loader):
#                 # Move tensors to GPU
#                 x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
#                 y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
                
#                 # Move pitch and energy to GPU
#                 pitch = pitch.cuda(rank, non_blocking=True)
#                 energy = energy.cuda(rank, non_blocking=True)

#                 # Evaluate Generator
#                 (
#                     (z, z_m, z_logs, logdet, z_mask),
#                     (x_m, x_logs, x_mask),
#                     (attn, logw, logw_),
#                     pred_pitch,
#                     pred_energy,
#                 ) = generator(x, x_lengths, y, y_lengths, pitch=pitch, energy=energy, gen=False)

#                 # Calculate losses
#                 l_mle = common_modified.mle_loss(z, z_m, z_logs, logdet, z_mask)
#                 l_length = common_modified.duration_loss(logw, logw_, x_lengths)
#                 l_pitch = common_modified.pitch_loss(pred_pitch, pitch)
#                 l_energy = common_modified.energy_loss(pred_energy, energy)

#                 loss_gs = [l_mle, l_length, l_pitch, l_energy]
#                 loss_g = sum(loss_gs)

#                 # Accumulate losses over the batches
#                 if batch_idx == 0:
#                     losses_tot = loss_gs
#                 else:
#                     losses_tot = [x + y for (x, y) in zip(losses_tot, loss_gs)]

#                 # Log progress
#                 if batch_idx % hps.train.log_interval == 0:
#                     logger.info(
#                         "Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
#                             epoch,
#                             batch_idx * len(x),
#                             len(val_loader.dataset),
#                             100.0 * batch_idx / len(val_loader),
#                             loss_g.item(),
#                         )
#                     )
#                     logger.info([x.item() for x in loss_gs])

#         # Normalize and log total loss
#         losses_tot = [x / len(val_loader) for x in losses_tot]
#         loss_tot = sum(losses_tot)

#         scalar_dict = {"loss/g/total": loss_tot}
#         scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_tot)})

#         # Write evaluation results
#         utils.summarize(
#             writer=writer_eval, global_step=global_step, scalars=scalar_dict
#         )

#         # Log epoch summary
#         logger.info("====> Epoch: {}".format(epoch))


if __name__ == "__main__":
    main()
