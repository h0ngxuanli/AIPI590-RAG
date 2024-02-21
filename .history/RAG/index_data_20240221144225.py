from tqdm import tqdm
import pandas as pd
from utils import cos_similarity
import torch
from pinecone import Pinecone, ServerlessSpec

def store_embedding(chunks_embedding, chunks, chunks_topic):
    pc = Pinecone(api_key='5a727c08-72c8-4031-8d55-7b268c06c443')
    pc.create_index(
        name="aipi590",
        dimension=512,
        metric="cosine",
        spec=ServerlessSpec(
            cloud = "gcp",
            region='us-central1'
        ) 
    )
        
    
    index = pc.Index("aipi590")
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
    embeddings_data = {"id":[], "values":[], "text":[]}
    embeddings = index.fetch([f'vec_{i}' for i in range(num_embed)])
    for i in range(num_embed):
        embeddings_data["id"].append(i)
        idx = f"vec_{i}"
        embeddings_data["text"].append(embeddings['vectors'][idx]['metadata']['text'])
        embeddings_data["values"].append(embeddings['vectors'][idx]['values'])
        
    return pd.DataFrame(embeddings_data)


def retrieve_content(query_embedding, corpus_embeddings, top_k):
    sim_score = cos_similarity(query_embedding, corpus_embeddings)
    
    scores, idxs = torch.topk(sim_score, top_k, dim=1)
    
    return {'corpus_id': idxs, 'score': scores}
