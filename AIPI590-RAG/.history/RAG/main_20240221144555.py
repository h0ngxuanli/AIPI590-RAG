import argparse
import pandas as pd
from dataset.mmlu_data import load_data
from RAG.embed_data import Encoder
from RAG.chunk_data import sliding_window
from RAG.index_data import store_embedding, retrieve_all_embedding, retrieve_content
from pinecone import Pinecone


def main():
    parser = argparse.ArgumentParser(description='aipi590')
    parser.add_argument('--model_path', type=str, default= "/home/featurize/work/Transformer-Patcher-main/model_cache/llama2-7b" )

    parser.add_argument('--max_seq_len', type=int, default=384)
    
    parser.add_argument('--chunk_size', type=int, default=300)
    parser.add_argument('--step_size', type=int, default=200)
    
    parser.add_argument('--top_k', type=int, default=3)

    parser.add_argument('--index_name', type=str, default="aipi590")
    
    
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')


    args = parser.parse_args()
    
    
    print("Loading MMLU machine learning dataset!")
    test_df, few_shots_df = load_data()
    

    print("Starting chunking wiki text data!")
    wiki_data = pd.read_parquet('./data/wiki_ml_mediawiki_cleaned.parquet').values.tolist()
    chunks, chunks_topic = sliding_window(wiki_data, args.chunck_size, args.step_size)


    print("Starting chunks embedding!")
    model = Encoder(args.model_path, args.max_seq_len)
    chunks_embedding = model.encode(chunks, show_progress_bar=True)
    
    
    print("Indexing chunks embedding!")
    if args.index_name is not None:
        pc = Pinecone(api_key='5a727c08-72c8-4031-8d55-7b268c06c443')
        index = pc.Index(args.index_name)
    else:
        index = store_embedding(chunks_embedding, chunks, chunks_topic)
    
    
    embedding_df = retrieve_all_embedding(index, len(chunks_embedding))
    
    
    print("Retireving relavant content!")
    corpus_embeddings = embedding_df["values"].tolist()
    
    
    results = retrieve_content(query_embedding, corpus_embeddings, top_k = args.top_k)
    
    
    

    

    # Get embeddings of chunks
    chunks_embedding = model.encode(chunks, show_progress_bar=True)
    
    
    
    
    
    
    
    
if __name__ == "__main__":
    main()