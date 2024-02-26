import argparse
import pandas as pd
from model.embedding_model import Encoder
from model.inference_model import LlaMA2
from dataset.mmlu_data import load_data
from get_response import retrieve_context, ask_question
import os
import sys
sys.path.insert(1, os.getcwd())




def main():
    parser = argparse.ArgumentParser(description='aipi590')
    parser.add_argument('--inference_model_path', type=str, default= "/home/featurize/work/Transformer-Patcher-main/model_cache/llama2-7b")
    parser.add_argument('--embedding_model_path', type=str, default= "BAAI/bge-large-en-v1.5")
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--top_k', type=int, default=2)
    args = parser.parse_args()
    
    
    embedding_model = Encoder(args.embedding_model_path, args.max_seq_len)
    inference_model = LlaMA2(args.inference_model_path)
    
    
    
    embedding_df = pd.read_parquet("./data/embeddings.parquet")
    test_df, few_shots_df = load_data()

    for i in range(len(test_df)):
        prompt = test_df.iloc[[i], :].input.values[0]
        a, b, c, d = test_df.iloc[[i], 1:-1].values[0]

        query = f"""
            Question: {prompt}\n
            A) {a}\n
            B) {b}\n
            C) {c}\n
            D) {d}\n
            """
        context = retrieve_context(query, embedding_df, embedding_model, args.top_k)
        answer = ask_question(query, context, inference_model)
        
        print(answer)
    
if __name__ == "__main__":
    main()