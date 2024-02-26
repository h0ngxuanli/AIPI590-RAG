from RAG.index_data import store_embedding, retrieve_all_embedding, retrieve_content
from pinecone import Pinecone
import pinecone


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
def retrieve(query, encoder, top_k)
    
    embedding_df = pd.read_parquet("/home/featurize/work/AIPI590-RAG/data/embeddings.parquet")
    
    
    corpus_embedding = embedding_df["embedding"].tolist()
    query_embedding = model.encode([query], show_progress_bar=False)
    sim_score = cos_similarity(query_embedding, corpus_embedding)
    scores, idxs = torch.topk(sim_score, top_k, dim=1)
    retrieve_context = embedding_df.loc[results['corpus_id'].numpy().flatten(), "text"].values
    context = " ".join([f"#{str(i)}" for i in retrieve_context])
    
    return context



def ask_question(query):
    
    context = retrieve(query, encoder, top_k)

    query = f"""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]

    Question: {query}

    Context: {context}

    ### Answer:\n
    """
    
    
    
    
    
    




#     print("Loading MMLU machine learning dataset!")
#     test_df, few_shots_df = load_data()
    
    
    
    
    
#     prompt = test_df.iloc[[0], :].input.values[0]
#     a, b, c, d = test_df.iloc[[0], 1:-1].values[0]

#     query = f"""Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]

#     Question: {prompt}\n
#     A) {a}\n
#     B) {b}\n
#     C) {c}\n
#     D) {d}\n

#     ### Answer:\n"""
    
    
    
#     # get 
#     results = retrieve_content(query_embedding, corpus_embeddings, top_k = args.top_k)