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

    ## I do not have enough time to store all the embeddings in pinecone...
    # embedding_df = retrieve_all_embedding(index, vector_count)
    # return embedding_df
    
    
# use cached embeddings in parquet file
def retrieve_context(query, embedding, model, top_k):
    """
    Retrieves the top-k most relevant contexts from a corpus based on a given query using cosine similarity.

    :param query: The query string for which context is to be retrieved.
    :param embedding: A pandas DataFrame containing the precomputed embeddings of the corpus with columns 'embedding' and 'text'.
    :param model: The model used to encode the query into an embedding.
    :param top_k: The number of top contexts to retrieve based on similarity.

    :return: A string concatenating the top-k contexts, each preceded by a '#' and separated by new lines.
    """
    
    # get embedding
    corpus_embedding = embedding["embedding"].tolist()
    query_embedding = model.encode([query], show_progress_bar=False)
    
    #compute sim score
    sim_score = cos_similarity(query_embedding, corpus_embedding)
    
    # retrieve context
    scores, corpus_id = torch.topk(sim_score, top_k, dim=1)
    retrieve_context = embedding.loc[corpus_id.numpy().flatten(), "text"].values
    
    # merge context into prompt
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



