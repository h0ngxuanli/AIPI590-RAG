import argparse
import pandas as pd
from model.embedding_model import Encoder
from model.inference_model import LlaMA2
from dataset.mmlu_data import load_data
from get_response import retrieve_context, ask_question
import numpy as np
from tqdm import tqdm
import os
import sys
import torch
import random
sys.path.insert(1, os.getcwd())

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(32)


def get_answer(
    embedding_model_path = "BAAI/bge-large-en-v1.5",
    inference_model_path = "/home/featurize/work/Transformer-Patcher-main/model_cache/llama2-7b",
    top_k = 2,
    question = None,
    ):


    
    # get embedding model and LLaMA-2-&B model
    embedding_model = Encoder(embedding_model_path, 512)
    inference_model = LlaMA2(inference_model_path)
    
    # load embeddings
    embedding_df = pd.read_parquet("./data/embeddings.parquet")
    test_df, few_shots_df = load_data()
    
    
    few_shots = []
    for i in range(3):
        prompt = few_shots_df.iloc[[i], :].input.values[0]
        a, b, c, d = few_shots_df.iloc[[i], 1:-1].values[0]
        answer = few_shots_df.iloc[[i], -1].values[0]
        question_shot = f"""
            Question: {prompt}\n
            A) {a}\n
            B) {b}\n
            C) {c}\n
            D) {d}\n
            ### Answer: {answer}\n
            """
        few_shots.append(question_shot)

    few_shots = " \n ".join(few_shots)


    if question is None:
        responses = []
        right_answers = []
        
        for i in tqdm(range(len(test_df))):
            
            
            prompt = test_df.iloc[[i], :].input.values[0]
            a, b, c, d = test_df.iloc[[i], 1:-1].values[0]
            right_answer = test_df.iloc[[i], -1].values[0]
            
            
            right_answers.append(right_answer)


            question = f"""
                Question: {prompt}\n
                A) {a}\n
                B) {b}\n
                C) {c}\n
                D) {d}\n
                ### Answer: \n
                """
            context = retrieve_context(question, embedding_df, embedding_model, top_k)
            answer = ask_question(question, few_shots, context, inference_model)
            responses.append(answer)
            
            if right_answer == answer:
                print(prompt)
                print(context)
            
        return (np.array(right_answers) == np.array(responses)).sum() / len(right_answers)
            
    else:
        context = retrieve_context(question, embedding_df, embedding_model, top_k)
        answer = ask_question(question, few_shots, context, inference_model)
        
        return answer, context
    
    
    