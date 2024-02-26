from tqdm import tqdm
import pandas as pd
from utils import cos_similarity
import torch
from pinecone import Pinecone, ServerlessSpec

def store_embedding(chunks_embedding, chunks, chunks_topic, index_name, API_key):
    
    """
    Stores embeddings with their associated text and topics into a Pinecone index.
    :return: The created Pinecone index object.
    """
    pc = Pinecone(api_key=API_key)
    
    pc.create_index(
        name=index_name,
        dimension=chunks_embedding.shape[1],
        metric="cosine",
        spec=ServerlessSpec(
            cloud = "gcp",
            region='us-central1'
        ) 
    )
    
    # index data int pinecone
    index = pc.Index(index_name)
    for i in tqdm(range(len(chunks_embedding))):
        index.upsert(
            vectors=[
                {
                    'id': f'vec_{i}',
                    'values': chunks_embedding[i],
                    'metadata': {"text":chunks[i], "topic":chunks_topic[i]}
                }
            ],
        )
    return index

def retrieve_all_embedding(index, num_embed):
    
    # retrieve embeddings from vector database
    embeddings_data = {"id":[], "values":[], "text":[]}
    embeddings = index.fetch([f'vec_{i}' for i in range(num_embed)])
    for i in range(num_embed):
        embeddings_data["id"].append(i)
        idx = f"vec_{i}"
        embeddings_data["text"].append(embeddings['vectors'][idx]['metadata']['text'])
        embeddings_data["values"].append(embeddings['vectors'][idx]['values'])
        
    return pd.DataFrame(embeddings_data)


# def retrieve_content(query_embedding, corpus_embeddings, top_k):
#     sim_score = cos_similarity(query_embedding, corpus_embeddings)
    
#     scores, idxs = torch.topk(sim_score, top_k, dim=1)
    
#     return {'corpus_id': idxs, 'score': scores}
