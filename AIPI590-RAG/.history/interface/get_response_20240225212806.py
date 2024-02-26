from RAG.utils import cos_similarity
from pinecone import Pinecone
import pinecone
import torch
import pandas as pd


# def retrieve(query, encoder, index_name,  API_key,  top_k)

    # retrive all embeddings from index to compute similarity
    
    # pc = Pinecone(api_key=API_key)
    # index = pc.Index(index_name)
    # stats = index.describe_index_stats()
    # namespace_map = stats['namespaces']
    # ret = []
    # for namespace in namespace_map:
    #     vector_count = namespace_map[namespace]['vector_count']

    # I do not have enough time to store all the embeddings in pinecone...
    # embedding_df = retrieve_all_embedding(index, vector_count)
    
    
# use cached embeddings in parquet file
def retrieve_context(query, embedding, model, top_k):
    
    
    # get 
    corpus_embedding = embedding["embedding"].tolist()
    query_embedding = model.encode([query], show_progress_bar=False)
    sim_score = cos_similarity(query_embedding, corpus_embedding)
    scores, corpus_id = torch.topk(sim_score, top_k, dim=1)
    retrieve_context = embedding.loc[corpus_id.numpy().flatten(), "text"].values
    context = " \n ".join([f"#{str(i)}" for i in retrieve_context])
    
    return context


def ask_question(question, few_shots, context, inference_model):
    
    query = f"""
    You are an assistant for machine learning Exame tasks. The retrieved context may help answer the question. Answer should only include a one word answer among [A, B, C, D].
    
    Examples: {few_shots}
    
    Context: {context}

    Questions: {question}
    """
    
    answer = inference_model.predict(query)
    
    return answer



