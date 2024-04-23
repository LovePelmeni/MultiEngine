import dataclasses
from src.multimodal.image_encoder import ImageEncoder
from src.multimodal.title_encoder import TitleEncoder
from src.multimodal.desc_encoder import DescriptionEncoder
from src.training.trainers import base
from torch.utils.data import dataset
from src.training.callbacks import (
    checkpoints,
    devices,
    early_stopping,
    logistics,
    network_state,
    distributed as call_dist
)
import numpy
import pathlib
from torch.distributed.optim import zero_redundancy_optimizer as zero
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
from tqdm import tqdm
import typing
from torch import nn
from torch.utils import data
from src.training.contrastive_learning import sampler
import torch
from torch import optim
from torch.optim import lr_scheduler
from torch import device
import pathlib
from src.metrics import metrics
from src.training.regularization import grad_blend

@dataclasses.dataclass
class TrainerConfig(object):
    """
    Configuration instance, that is responsible
    for training single network instance.
    """
    network: nn.Module
    train_devices: typing.List[torch.DeviceObjType]
    optimizer_config: typing.Dict[str, typing.Any]
    lr_scheduler_config: typing.Dict[str, typing.Any]

@dataclasses.dataclass
class SnapshotConfig(object):
    """
    Configuration class for the training checkpointing
    of the network.

    Parameters:
    -----------
        save_every: number of epochs before making network checkpoint.
        snapshot_log_dir - folder directory to store snapshot files.
        snapshot_ext - extention to use for the snapshot file.
    """
    snapshot_path_name: str
    snapshot_ext: typing.Literal['pt', 'pth'] = 'pt'


class ContrastiveTrainer(base.BaseTrainer):
    """
    Training pipeline for contrastive learning
    of multiple embedding generation networks:

        - Autoencoder-backed image embedding generator for processing image data.
        - DistilBERT-backed word embedding generator for processing text data.

    Parameters:
    -----------
        train_configs - typing.List[TrainerConfig]: list of embedding generation networks for each modality.
        contrastive_sampler - typing.Callable - : sampler for hard mining sample pairs.
        batch_size: (int) - size of the data batch, feed to networks at each iteration
        pair_loss_name (str)- loss function to use for contrastive learning. (matching between hard and soft samples of single modality)
        modal_loss_name (str)- loss function to compare embeddings similarity between multiple modalities.
        eval_metric_name (str)- evaluation metric for assessing similarity between multimodal embeddings. (typical choice: Levenstein Distance, Cosine Similarity)
        log_dir (str)- main directory for storing logging statistics
        distributed: bool - enable single machine distributed training.
        dist_rank (int)- global rank for single machine distributed training.
        dist_backend (str) - backend for accelerating distibuted training.
        world_size (int) - number of GPUs participating in distributed training.
        group_name (str)- distributed training group name.
        reproducible (bool) - fixate experiment random settings.
    """
    def __init__(self,
        train_configs: typing.List[TrainerConfig],
        snapshot_configs: typing.List[SnapshotConfig],
        contrastive_sampler: sampler.BaseSampler,
        batch_size: int,
        pair_loss_name: str,
        modal_loss_name: str,
        eval_metric_name: str,
        log_dir: typing.Union[str, pathlib.Path],
        distributed: bool = False,
        dist_rank: int = None,
        dist_backend: typing.Literal["nccl", "golo"] = None,
        world_size: int = None,
        group_name: str = None,
        reproducible: bool = False
    ):
        super(ContrastiveTrainer, self).__init__()

        self.contrastive_sampler = contrastive_sampler
        self.batch_size = batch_size
        self.distributed = distributed 
        self.reproducible = reproducible
        self.world_size = world_size 
        self.dist_rank = dist_rank 
        self.dist_backend = dist_backend
        self.group_name = group_name

        self.configure_callbacks(
            base_log_dir=log_dir, 
            snapshot_configs=snapshot_configs
        )

        self.on_init_start()
        self.configure_setup(train_configs=train_configs)

        # two loss functions for contrasive learning training.
        # 1. loss for measuring similarity / dissimilarity between hard negative and positive pairs
        # 2. loss for measuring similarity between embeddings from multiple modalities

        self.pair_loss_function = self.load_loss(pair_loss_name)
        self.modal_loss_function = self.load_loss(modal_loss_name)
        self.eval_metric = self.load_metric(eval_metric_name)
        self.save_every: int = save_every

        # OGR settings for joint multimodal training.
        self.blend_regularizer = grad_blend.GradientBlending()

    def configure_setup(self, train_configs: typing.List[TrainerConfig]):
        """
        Configures networks, optimization algorithms and
        learning rate schedulers.
        
        Parameters:
        -----------
            train_configs - list of TrainerConfig objects
        """
        self.networks = []
        self.optimizers = []
        self.schedulers = []

        for config in train_configs:

            network = self.configure_network(
                network=config.network,
                device_ids=config.train_devices,
                output_device=config.output_device
            )
            optimizer = self.configure_optimizer(
                network=network,
                optimizer_config=config.optimizer_config
            )
            lr_scheduler = self.configure_lr_scheduler(
                optimizer=optimizer,
                lr_scheduler_config=config.lr_scheduler_config
            )
            self.networks.append(network)
            self.optimizers.append(optimizer)
            self.schedulers.append(lr_scheduler)

    def configure_snapshot_callbacks(self, 
        base_log_dir: str, 
        snapshot_configs: typing.List[SnapshotConfig]):
        """
        Configures snapshoting for each
        modality encoder.

        Parameters:
        -----------
            base_log_dir - base directory to store snapshots
            snapshot_configs - list of snapshot configurations.
        """
        if len(snapshot_configs) == 0:
            return 

        for config in snapshot_configs:

            snap_path_name = config.snapshot_path_name
            snap_ext = config.snapshot_ext
            snapshot_log_dir = os.path.join(base_log_dir, snap_ext)

            self.callbacks.append(
                checkpoints.SnapshotCallback(
                    snapshot_ext=snap_ext, 
                    save_every=self.save_every, 
                    log_dir=snapshot_log_dir
                )
            )
        
    def configure_callbacks(self, 
        snapshot_configs: typing.List[SnapshotConfig], 
        base_log_dir: typing.Union[str, pathlib.Path]):

        report_log_dir = os.path.join(base_log_dir, "reports")
        cpu_log_dir = os.path.join(base_log_dir, "cpu")
        gpu_log_dir = os.path.join(base_log_dir, "gpu")
        network_monitor_log_dir = os.path.join(base_log_dir, "network_health")
        
        self.callbacks = [
            logistics.LogisticsCallback(log_dir=report_log_dir),
            devices.CPUInferenceCallback(log_dir=cpu_log_dir),
            devices.GPUInferenceCallback(log_dir=gpu_log_dir),

            network_state.NetworkMonitoringCallback(
                log_dir=network_monitor_log_dir,
                weight_param_tag='weights',
                bias_param_tag='biases'
            )
        ]
         
        # configuring snapshot callbacks
        self.configure_snapshot_callbacks(base_log_dir, snapshot_configs)

        if self.distributed:
            dist_callback = call_dist.DistributedTrainCallback(
                rank=self.dist_rank,
                backend=self.dist_backend,
                world_size=self.world_size,
                group_name=self.group_name
            )
            self.callbacks.append(dist_callback)

    def configure_seed(self, input_seed: int):
        """
        Set network behaviour to be deterministic,
        including data loading, etc.
        Warning:
            do not use this method during training,
            it's main purpose lies in ability
            to provide an option for debugging tasks
            and may dramatically slow down training speed.
        """
        self.seed_generator = torch.manual_seed(seed=input_seed)
        random.seed(a=input_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def configure_loader(self, dataset: dataset.Dataset):

        if not self.distributed:
            return data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
            )
        else:
            if (self.reproducible == True):
                return data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    generator=self.seed_generator,
                    pin_memory=True
                )
            else:
                return data.DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
    
    def predict_embs(self, 
        data_sample: typing.Tuple[
            torch.Tensor, 
            torch.Tensor, 
            torch.Tensor
        ]) -> torch.Tensor:
        embs = []
        for idx, modality in enumerate(data_sample):
            pred_emb = self.networks[idx].forward(modality)
            embs.append(pred_emb)
        return embs

    def train(self, train_dataset: dataset.Dataset):

        self.on_init_start()

        global_step = 0
        curr_metric = 0

        image_overall_loss = float('inf')
        desc_overall_loss = float('inf')
        title_overall_loss = float('inf')
        img_title_modal_loss = float('inf')
        img_desc_modal_loss = float('inf')

        # turning multimodal encoders to
        # training mode.
        for net in self.networks:
            net.train()

        loader = self.configure_loader(train_dataset)
        self.on_init_end()

        weights = torch.ones((5,))

        for epoch in range(self.max_epochs):

            image_epoch_losses = []
            desc_epoch_losses = []
            title_epoch_losses = []
            img_title_modal_losses = []
            img_desc_modal_losses = [] 

            for images, titles, descriptions, labels in tqdm(loader, desc="""
                    epoch: %s; image_loss: %s; 
                    desc_loss: %s; title_loss: %s; 
                    curr_metric: %s;""" % (
                        epoch, image_overall_loss, 
                        desc_overall_loss, title_overall_loss, 
                        curr_metric)):

                # finding hard pairs of (pos_sample, sample, neg_sample) for
                # contrastive learning training, using current batch

                image_hard_pairs = self.contrastive_sampler.hard_mining(
                    batch_data=images, 
                    batch_labels=labels,
                    data_type='image'
                )

                title_hard_pairs = self.contrastive_sampler.hard_mining(
                    batch_data=titles,
                    batch_labels=labels,
                    data_type='text'
                )

                description_hard_pairs = self.contrastive_sampler.hard_mining(
                    batch_data=descriptions,
                    batch_labels=labels,
                    data_type='text'
                )

                # merge hard pairs from multiple modalities together
                zipped_pairs = zip(
                    image_hard_pairs, 
                    text_hard_pairs, 
                    descriptions_hard_pairs
                )
        
                # each pair follow format: (image, text), for both positive, main and negative.

                for (
                    pos_img_sample, 
                    img_sample, 
                    neg_img_sample
                    ), (
                        pos_title_sample, 
                        title_sample, 
                        neg_title_sample
                        ), (
                            pos_des_sample, 
                            desc_sample, 
                            neg_desc_sample) in zipped_pairs:
                    
                    self.on_train_batch_start()
        
                    pos_v_emb, pos_t_emb, pos_d_emb = self.predict_embs([
                        pos_img_sample, 
                        pos_title_sample, 
                        pos_desc_sample
                    ])

                    pair_v_emb, pair_t_emb, pair_d_emb = self.predict_embs([
                        img_sample, 
                        title_sample, 
                        desc_sample
                    ])

                    neg_v_emb, neg_t_emb, neg_d_emb = self.predict_embs(
                        [neg_img_sample, 
                        neg_title_sample, 
                        neg_desc_sample]
                    )
        
                    img_loss = self.pair_loss_function(pos_v_emb, pair_v_emb, neg_v_emb)
                    desc_loss = self.pair_loss_function(pos_d_emb, pair_d_emb, neg_d_emb)
                    title_loss = self.pair_loss_function(pos_t_emb, pair_t_emb, neg_t_emb)
                    
                    img_title_modal_sim_loss = self.modal_loss_function(pair_v_emb, pair_t_emb)
                    img_desc_modal_sim_loss = self.modal_loss_function(pair_v_emb, pair_d_emb)
                    
                    # overall loss function: summary of image similarity pairs, text similarity pairs
                    # and similarity between modalities

                    img_encoder_loss = weights[0] * img_loss + weights[3] * img_title_modal_sim_loss
                    title_encoder_loss = weights[1] * title_loss + weights[3] * img_title_modal_sim_loss
                    desc_encoder_loss = weights[2] * desc_loss + weights[4] * img_desc_modal_sim_loss
                    
                    image_epoch_losses.append(img_encoder_loss.item())
                    title_epoch_losses.append(title_encoder_loss.item())
                    desc_epoch_losses.append(desc_encoder_loss.item())
                    
                    # in case we are using single gpu, we traverse
                    # over all computed loss (for each modality) and after each update
                    # clear gradients

                    for idx in range(len(self.optimizers)):
                        
                        if isinstance(self.networks[idx], ImageEncoder):
                            img_encoder_loss.backward()

                        elif isinstance(self.networks[idx], TitleEncoder):
                            text_encoder_loss.backward()
                        
                        elif isinstance(self.networks[idx], DescriptionEncoder):
                            desc_encoder_loss.backward()
                    
                        self.optimizers[idx].step()

                        if len(self.lr_schedulers) >= (idx+1):
                            self.lr_schedulers[idx].step()

                        # # emptying the gradients, so they does not overlap
                        # # with next ones, when training multiple networks
                        # # on the same device.
                        self.optimizers[idx].zero_grad()

                    self.on_train_batch_end()

            image_overall_loss = numpy.mean(image_epoch_losses)
            title_overall_loss = numpy.mean(title_epoch_losses)
            desc_overall_loss = numpy.mean(desc_overall_losses)
            img_title_modal_loss = numpy.mean(img_title_modal_losses)
            img_desc_modal_loss = numpy.mean(img_desc_modal_losses)


            # updating Gradient Blending weights

            if (epoch + 1) % self.save_every == 0:
                train_losses = torch.FloatTensor(
                    [
                        image_overall_loss, 
                        title_overall_loss, 
                        desc_overall_loss
                    ], 
                    requires_grad=False
                )

                validation_losses = torch.FloatTensor(
                    [
                        image_validation_loss,
                        title_validation_loss,
                        desc_validation_loss
                    ], 
                    requires_grad=False
                )

                weights = self.blend_regularizer.compute_weights(
                    curr_train_losses=train_losses, 
                    curr_valid_losses=validation_losses
                )

            # we pass argument 'trainer' to this event
            # in case early stopping callback want to say us, that training is done.
            # It will update flag 'stop' to True
            self.on_train_epoch_end(trainer=self, overall_loss=overall_loss)
            
            # global step is simply used to track current epoch.
            self.on_validation_start(global_step=global_step)
            self.on_validation_end(global_step=global_step)
            self.on_train_epoch_end(global_step=global_step)
            
            if self.stop: break

        self.tearDown()

    def find_similarity(self, embeddings_group):
        """
        Finds similarity between embeddings in a given
        group.
        Parameters:
        -----------
            embeddings_group - list of similar embeddings to compare
        """
        total_sim = []
        for emb1 in range(len(embeddings_group)):
            emb_sims = []
            for emb2 in range(emb1, len(embeddings_group)):
                sim = (
                    torch.dot(embeddings_group[emb1], embeddings_group[emb2])
                    ) / (
                        torch.norm(embeddings_group[emb1]) 
                        * torch.norm(embeddings_group[emb2])
                    )
                emb_sims.append(sim.item())
            total_sim.append(numpy.mean(emb_sims))
        return numpy.mean(total_sim)

    def sliced_evaluate(self, embeddings: typing.List[torch.Tensor], labels: typing.List):
        """
        Evaluates embeddings on individual slices of data,
        based on the label.
        """
        output_metrics: typing.Dict[str, float] = {}
        unique_labels = numpy.unique(labels)
        for label in unique_labels:
            indices = numpy.where(labels == label)[0]
            cat_embeddings = [emb for emb in embeddings if emb in indices]
            metric = self.find_similarity(cat_embeddings)
            output_metrics[label] = metric
        return output_metrics
