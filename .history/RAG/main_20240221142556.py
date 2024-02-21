import argparse

from dataset.mmlu_data import load_data
from RAG.embed_data import Encoder
from RAG.chunk_data import sliding_window

def main():
    parser = argparse.ArgumentParser(description='aipi590')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')


    args = parser.parse_args()
    
    
    
    test_df, few_shots_df = load_data()
    
    
    
    model = Encoder()
    
    
    
    store_embedding(chunks_embedding, chunks, chunks_topic)
    
    
if __name__ == "__main__":
    main()