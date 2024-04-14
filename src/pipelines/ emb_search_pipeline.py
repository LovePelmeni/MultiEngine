"""
Pipeline for generating embedding vectors
of articles.
The purpose is to create dataset for training
similarity search algorithm, which will cluster the objects
into meaningful groups, so we can generate recommendations, based
on the incoming data.
"""

def main():
    parser = argparse.ArgumentParser(description="pipeline for generating article embeddings")

    emb_gen_group = parser.add_group("gen_network_group", type=str)
    emb_gen_group.add_argument("--description-encoder-config-path", type=str, help='configuration for description modality encoder')
    emb_gen_group.add_argument("--title-encoder-config-path", type=str, help='configuration for title modality encoder')
    emb_gen_group.add_argument("--image-encoder-config-path", type=str, help='configuration for image modality encoder')

    inf_device_group = parser.add_mutual_exclusive_group("inference device group")
    inf_device_group.add_argument("--use-cpu", type=bool, action=False, help='use CPU for inference', default=True)
    inf_device_group.add_argument("--use-gpu", type=bool, action=True, help='use GPU for inference', default=False)
    inf_device_group.add_argument("--use-mps", type=bool, action=True, help='use MPS for inference', default=False)

    output_data_group = parser.add_group("ouput_settings_group")

    output_data_group.add_argument("--output-emb-dataset-path", 
        type=str, 
        help='path to the embedding dataset, where generated embeddings will be stored.'
    )

    output_data_group.add_argument(
        "--output-emb-dataset-dtype", 
        type=str, 
        help='data type for storing generated embeddings'
    )

    output_data_group.add_argument(
        "--output-emb-metadata-path", 
        type=str, 
        help='path to the folder, where metadata about generated embeddings will be stored.'
    )

    output_data_group.add_argument(
        "--output-emb-metadata-dtype", 
        type=str, 
        choices=[],
        help='data type to use for storing metadata'
    )

    extra_group = parser.add_group("extra_arguments_group")
    extra_group.add_argument("--log-dir")
    extra_group.add_argument("--experiment-prefix")
    
if __name__ == '__main__':
    main()
