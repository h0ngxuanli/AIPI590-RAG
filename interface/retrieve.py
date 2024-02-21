

from RAG.index_data import store_embedding, retrieve_all_embedding, retrieve_content


    # retrive all embeddings from index to compute similarity
    embedding_df = retrieve_all_embedding(index, len(chunks_embedding))
    
    
    print("Retireving relavant content!")
    # use cosine similarity to retrieve text
    corpus_embeddings = embedding_df["values"].tolist()
    
    
    
    print("Loading MMLU machine learning dataset!")
    test_df, few_shots_df = load_data()
    
    
    
    
    
    prompt = test_df.iloc[[0], :].input.values[0]
    a, b, c, d = test_df.iloc[[0], 1:-1].values[0]

    query = f"""Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]

    Question: {prompt}\n
    A) {a}\n
    B) {b}\n
    C) {c}\n
    D) {d}\n

    ### Answer:\n"""
    
    
    
    # get 
    results = retrieve_content(query_embedding, corpus_embeddings, top_k = args.top_k)