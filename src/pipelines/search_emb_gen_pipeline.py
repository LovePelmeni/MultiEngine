"""
Pipeline for generating embedding vectors
of articles.
The purpose is to create dataset for training
similarity search algorithm, which will cluster the objects
into meaningful groups, so we can generate recommendations, based
on the incoming data.
"""


"""
TODO:
    1. analyze which information (metadata)
    could be added to the embeddings, then
    create corrsponding dataframe and save it.
    2. create shell script for automatic pipeline execution.
"""

import pandas
import pathlib
import os
import argparse
from src.training.datasets import datasets
from torch.utils.tensorboard.summary_writer import (
    SummaryWriter
)
from pipelines.utils import (
    load_images,
    load_titles,
    load_descriptions
)
from functools import partial
import torch.multiprocessing as mp

def compute_embedding(
    network: nn.Module, 
    input_data: torch.Tensor, 
    inference_device: torch.device,
    queue: nn.Module
) -> torch.Tensor:
    """
    Function, that simply computes embedding vector 
    using given network and input data.
    """
    device_data = input_data.to(inf_device)
    output_emb = netowrk.to(inf_device).cpu()
    queue.put(output_emb)
    return output_emb

def main():
    parser = argparse.ArgumentParser(description="pipeline for generating article embeddings")

    dataset_group = parser.add_group("dataset_group")
    dataset_group.add_argument(
        "--dataset-type", 
        type=str, required=True, 
        choices=['train', 'valid'], 
        help='type of the input dataset'
    )
    dataset_group.add_argument(
        "--image-dataset-dir", 
        type=str, required=True,
        dest="image_dataset_dir",
        help='folder, where image modality is stored'
    )

    dataset_group.add_argument(
        "--title-dataset-dir",
        type=str, dest="title_dataset_dir",
        help='folder, containing textual title modality .txt documents'
    )

    dataset_group.add_argument(
        "--desc-dataset-dir",
        type=str, dest='desc_dataset_dir',
        help='folder, containing textual description modality .txt documents'
    )

    emb_gen_group = parser.add_group("gen_network_group")

    emb_gen_group.add_argument(
        "--description-encoder-config-path", 
        type=str, 
        dest='description_encoder_config_path'
        help='configuration for description modality encoder'
    )
    emb_gen_group.add_argument(
        "--title-encoder-config-path", 
        type=str, 
        dest='title_encoder_config_path',
        help='configuration for title modality encoder'
    )
    emb_gen_group.add_argument(
        "--image-encoder-config-path", 
        type=str, 
        dest='image_encoder_config_path',
        help='configuration for image modality encoder'
    )

    inf_device_group = parser.add_mutual_exclusive_group("inference device group")
    inf_device_group.add_argument("--use-cpu", type=bool, dest='use_cpu', action=False, help='use CPU for inference', default=True)
    inf_device_group.add_argument("--use-gpu", type=bool, dest='use_gpu', action=True, help='use GPU for inference', default=False)
    inf_device_group.add_argument("--use-mps", type=bool, dest='use_mps', action=True, help='use MPS for inference', default=False)

    output_data_group = parser.add_group("output_settings_group")

    output_data_group.add_argument("--output-emb-dataset-path", 
        type=str, 
        dest='output_emb_dataset_path',
        help='path to the embedding dataset, where generated embeddings will be stored.'
    )

    output_data_group.add_argument(
        "--output-emb-dataset-dtype", 
        type=str, 
        dest='output_emb_dataset_dtype',
        help='data type for storing generated embeddings'
    )

    output_data_group.add_argument(
        "--output-emb-metadata-path", 
        type=str, 
        dest='output_emb_metadata_path',
        help='path to the folder, where metadata about generated embeddings will be stored.'
    )

    extra_group = parser.add_group("extra_arguments_group")
    extra_group.add_argument("--log-dir", type=str, required=True, help='directory to store logs')
    extra_group.add_argument("--experiment-prefix", type=str, required=True, help='experiment unique ID')

    args = parser.parse_args()
    
    # setting up directories
    log_dir = pathlib.Path(args.log_dir)

    output_image_emb_dir = pathlib.Path(args.output_image_emb_dir)
    output_title_emb_dir = pathlib.Path(args.output_title_emb_dir)
    output_desc_emb_dir = pathlib.Path(args.output_desc_emb_dir)
    outptu_label_emb_filepath = pathlib.Path(args.output_label_emb_filepath)
 
    output_emb_metadata_dir = pathlib.Path(args.output_emb_metadata_path)

    input_image_dir = pathlib.Path(args.input_image_dataset_dir)
    input_title_dir = pathlib.Path(args.input_title_dataset_dir)
    input_desc_dir = pathlib.Path(args.input_desc_dataset_dir)

    # creating directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(output_emb_dataset_dir.parent, exist_ok=True)
    os.makedirs(output_emb_metadata_dir.parent, exist_ok=True)

    # loading data from multiple source modalities
    
    image_paths = load_image_paths(base_path=input_image_dir)
    title_paths = load_title_paths(base_path=input_title_dir)
    description_paths = load_description_paths(base_path=input_desc_dir)
    
    metadata_df = pandas.DataFrame(columns=['emb_index', 'label'])
    inference_device = torch.device('cpu')

    if (args.use_gpu == True):
        inference_device = torch.device("cuda")
    
    elif (args.use_mps == True):
        if torch.backends.mps.is_available():
            print("enabling mps inference.")
            inference_device = torch.device("mps")
        else:
            print("mps inference is not avaiable, switching to CPU")

    # loading encoders
    image_encoder = torch.load(args.image_encoder_config_path).to(inference_device)
    title_encoder = torch.load(args.title_encoder_config_path).to(inference_device)
    desc_encoder = torch.load(args.desc_encoder_config_path).to(inference_device)

    # turning encoders to evaluation mode
    image_encoder.eval()
    title_encoder.eval()
    desc_encoder.eval()

    # dataset for generating embeddings
    embedding_dataset = datasets.ContrastiveDataset(
        image_paths=image_paths,
        title_paths=title_paths,
        description_paths=description_paths,
        image_transformations=image_augmentations,
        title_transformations=title_augmentations,
        description_transformations=description_augmentations,
        labels=labels,
        dataset_type=dataset_type
    )
    # setting up data loader
    loader = data.DataLoader(
        dataset=embedding_dataset,
        batch_size=32,
        num_workers=num_workers,
        shuffle=True
    )

    image_embeddings = []
    title_embeddings = []
    description_embeddings = []
    labels = []

    if inference_device.lower() == "cpu":
        queue = mp.Queue()

    # generating embeddings
    for images, token_descriptions, token_titles, labels in loader:

        # NOTE:
        # token descriptions and token titles are tokenized sentences (pieces of text)
        # you don't need to run them through tokenizer again.

        device_images = images.to(inference_device)
        device_descriptions = token_descriptions.to(inference_device)
        device_titles = token_titles.to(inference_device)

        output_embeddings = []

        # if inference device was selected to be a CPU,
        # we split the computations into multiple streams using torch.multiprocessing library.
        # each process operates on a single data modality independently and returns output
        # back on the cpu

        if inference_device == 'cpu':

            process_units = [
                (
                    image_encoder.to(inference_device), 
                    image_embs
                ),
                (
                    title_encoder,
                    token_titles
                ),
                (
                    desc_encoder,
                    token_descriptions
                )
            ]
            
            for (process_encoder, process_data) in process_units:

                process_unit = mp.Process(
                    func=compute_embeddings, 
                    args=(process_encoder, process_data, inference_device, queue)
                )
                process_unit.start()
                processes.append(process)
            
            for idx, process in enumerate(multiprocesses):
                # getting result from the single CPU process
                output_emb = queue.get()

                if idx == 0:
                    image_embeddings.append(output_emb)

                if idx == 1:
                    title_embeddings.append(output_emb)

                if idx == 2:
                    desc_embeddings.append(output_emb)
                # completing process
                process.join()
        else:
            image_embs = image_encoder.to(inference_device).forward(device_images).cpu()
            title_embs = title_encoder.to(inference_device).forward(device_titles).cpu()
            desc_embs = desc_encoder.to(inference_device).forward(device_descriptions).cpu()
            
            image_embeddings = torch.stack([image_embeddings, image_embs], axis=0)
            text_embeddings = torch.stack([text_embeddings, text_embs], axis=0)
            description_embeddings = torch.stack([description_embeddings, desc_embs], axis=0)

        labels.extend(labels)

    # converting data to numpy representation.
    
    image_embeddings = image_embeddings.numpy()
    title_embeddings = title_embeddings.numpy()
    description_embeddings = description_embeddings.numpy()
    labels = numpy.asarray(labels)
    
    # saving embeddings and corresponding label metadata

    image_embeddings.to_file(output_image_emb_dir)
    title_embedddings.to_file(output_title_emb_dir)
    description_embedddings.to_file(output_desc_emb_dir)
    labels.to_file(output_label_emb_dir)
    
if __name__ == '__main__':
    main()
