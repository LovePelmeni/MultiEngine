from this import d
import numpy
import typing
import pathlib
import logging
import os

Logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='search_dataset_logs.log')
Logger.addHandler(file_handler)

class SearchVectorDataset(object):
    """
    Base module, responsible for manipulating / interacting
    with similarity search dataset to access vectors
    by indexes and other purposeful tasks.

    Parameters:
    -----------
        "dataset_path" (str) - path to .dat or .bin file, 
        containing structured vector dataset, 
        which was used for training similarity search index.

        "data_type" - numpy dtype representation to use when loading the data in memory
        "access_mode" - mode to open file with.
        "data_shape" - shape to use for reshaping your data.
    """
    def __init__(self, 
        dataset_path: typing.Union[str, pathlib.Path],
        access_mode: typing.Literal["r", "rb", "wb", "w"],
        data_shape: typing.Tuple,
        data_type: numpy.dtype, 
    ):
        ext = os.path.splitext(
            os.path.basename(dataset_path)
        )
        if ext.lower() not in ("dat", "bin"):
            raise RuntimeError(
                """extension: '%s' file is incompatible with numpy.memmap, \
                use either '.dat' or '.bin' files to store your data""")
        try:
            self.dataset_path = dataset_path
            self._mem_vec_data = numpy.memmap(
                filename=dataset_path, 
                dtype=data_type, 
                mode=access_mode,
                shape=data_shape
            )
            self.data_shape = data_shape
        except(FileNotFoundError):
            raise RuntimeError('dataset file is not found')

        except(Exception):
            raise Exception

    def add_vectors(self, embeddings: typing.List[numpy.ndarray]):
        """
        Adds new vectors to existing .dat storage.
        
        Parameters:
        -----------
            embeddings - list of numpy.ndarray embedding vectors.
        """
        try:
            if not all([emb.shape == self.data_shape for emb in embeddings]):
                for idx, emb in enumerate(embeddings):
                    if emb.shape != (1, self.data_shape.shape[1]):
                        embeddings[idx] = emb.reshape((1, self.data_shape[1]))
        
            embeddings: numpy.ndarray = numpy.stack(embeddings)
            with open(self.dataset_path, mode='ab') as dat_dataset_file:
                embeddings.tofile(fid=dat_dataset_file)
        except(FileNotFoundError) as err:
            Logger.debug("dataset does not exist under folder: '%s'" % self.dataset_path)

    def vectors_by_indices(self, indices: typing.List):
        try:
            return self._mem_vec_data[indices]
        except(Exception, IndexError) as err:
            Logger.error(err)
            return []