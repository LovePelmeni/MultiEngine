from src.training.callbacks import base
from torch import distributed
import logging
import typing
import uuid

logger = logging.getLogger(__name__)


class DistributedTrainCallback(base.BaseCallback):
    """
    Manage necessary components for distributed
    training of the network on multiple GPUs.
    """
    def __init__(self, 
        rank: int, 
        backend: typing.Literal["nccl", "golo"], 
        world_size: int,
        group_name: typing.Optional[str] = None
    ):
        self.rank = rank
        self.backend = backend
        self.world_size = world_size
        self.group_name = group_name

        if not group_name:
            self.group_name = str(uuid.uuid5()) + "_group_process"

    def on_init_start(self):
        try:
            distributed.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self.rank,
                group_name=self.group_name
            )
        except(Exception) as err:
            logger.error(err)

    def tearDown(self):
        try:
            distributed.destroy_process_group(group=self.pr_group_name)
        except(Exception) as err:
            logger.error(err)
    

