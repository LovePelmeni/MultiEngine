import numpy
import typing
import pathlib
import logging
import os
import pandas
import json

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
        "meta_decode_format" - format to use for JSON metadata string when it is saved to the database file.
        Example:
            json_str_metadata  = json.dumps({'name': 'Vlad'})
            enc_json_string = json_str_metadata.encode(encode_format) <----- this is the decode_format
            ...... saving to database
    """
    def __init__(self, 
        emb_dataset_path: typing.Union[str, pathlib.Path],
        meta_dataset_path: typing.Union[str, pathlib.Path],
        access_mode: typing.Literal["r", "rb", "wb", "w"],
        emb_data_shape: typing.Tuple,
        meta_data_shape: typing.Tuple,
        emb_data_type: numpy.dtype, 
        meta_data_type: numpy.dtype,
        meta_decode_format: typing.Literal['utf-8', 'utf-16', 'utf-32'] = 'utf-8'
    ):
        emb_ext = os.path.splitext(
            os.path.basename(emb_dataset_path)
        )
        label_ext = os.path.splitext(
            os.path.basename(emb_dataset_path)
        )
    
        if emb_ext.lower() not in ("dat", "bin", "fvecs") or label_ext not in ("dat", "bin", "fvecs"):
            raise RuntimeError(
                """extension: '%s' file is incompatible with numpy.memmap, \
                use either '.dat' or '.bin' files to store your data""")
        try:
            self.emb_dataset_path = emb_dataset_path
            self.label_dataset_path = meta_dataset_path

            self._mem_vec_data_metadata: pandas.DataFrame = self.load_metadata_dataset(
                meta_dataset_path,
                meta_data_type,
                access_mode,
                meta_data_shape,
                meta_decode_format
            )
            self._mem_vec_data: numpy.ndarray = self.load_product_embedding_dataset(
                emb_dataset_path=emb_dataset_path,
                emb_data_type=emb_data_type,
                emb_data_shape=emb_data_shape,
                access_mode=access_mode
            )
            self.label_data_shape = meta_data_shape
            self.emb_data_shape = emb_data_shape

            self.label_data_type = meta_data_type 
            self.emb_data_type = emb_data_type 

        except(FileNotFoundError):
            raise RuntimeError('dataset file is not found')

        except(Exception):
            raise Exception

    def load_product_embedding_dataset(self,
        emb_dataset_path: str,
        emb_data_type: numpy.dtype,
        emb_data_shape: tuple,
        access_mode: str = 'r',
    ) -> numpy.ndarray:
        """
        Loads product embedding data from ther remote
        dataset file.
        """
        return numpy.memmap(
            filename=emb_dataset_path,
            dtype=emb_data_type,
            mode=access_mode,
            shape=emb_data_shape
        )

    def load_metadata_dataset(self, 
        meta_dataset_path: str, 
        meta_data_type: numpy.dtype, 
        access_mode: str, 
        meta_data_shape: tuple,
        decode_format: str

    ) -> pandas.DataFrame:
        """
        Loads metadata information about embedding
        vectors from the remote file.
        """
        try:
            loaded_dataset = numpy.memmap(
                filename=meta_dataset_path,
                dtype=meta_data_type,
                mode=access_mode,
                shape=meta_data_shape
            )
            for idx, row in enumerate(loaded_dataset):
                loaded_dataset[idx] = json.loads(
                    row.decode(decode_format)
                )
            # creating pandas DataFrame of metadata records
            metadata_props = list(loaded_dataset[0].keys())
            meta_df = pandas.DataFrame({
                prop: [data.get(prop, None) for data in loaded_dataset]
                for prop in metadata_props
            })
            return meta_df
            
        except(Exception) as err:
            Logger.critical(err)
            raise RuntimeError("failed to load embedding metadata")

    def add_vectors(self, embeddings: typing.List[numpy.ndarray], labels: typing.List[str]):
        """
        Adds new vectors to existing .dat storage.
        
        Parameters:
        -----------
            embeddings - list of numpy.ndarray embedding vectors.
            labels - corresponding list of JSON encoded objects, representing information about 
            each individual embedding.
        """
        try:
            out_embeddings = numpy.zeros(shape=(len(embeddings)), dtype=self.emb_data_type)
            out_labels = numpy.zeros(shape=(len(labels),), dtype=self.label_data_type)

            if not all([emb.shape == self.data_shape for emb in embeddings]):
                for idx, emb in enumerate(embeddings):
                    if emb.shape != (1, self.data_shape.shape[1]):
                        out_embeddings[idx] = emb.reshape((1, self.data_shape[1]))
                        out_labels[idx] = labels[idx]
        
            out_embeddings: numpy.ndarray = numpy.stack(embeddings)

            with open(self.emb_dataset_path, mode=self.access_mode) as dat_emb_file:
                out_embeddings.tofile(fid=dat_emb_file)

            with open(self.label_dataset_path, mode=self.access_mode) as dat_label_file:
                out_labels.tofile(fid=dat_label_file)

        except(FileNotFoundError):
            Logger.debug("dataset does not exist under folder: '%s'" % self.dataset_path)

    def __getitem__(self, idx: int):
        try:
            info = self._mem_vec_data_metadata.iloc[idx]
            embedding = self._mem_vec_data[idx]
            return info, embedding
            
        except(Exception, IndexError) as err:
            Logger.error(err)
            return None, None
