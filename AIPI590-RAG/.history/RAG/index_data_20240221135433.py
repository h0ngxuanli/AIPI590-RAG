from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm





def vector_store(chunks_embedding, chunks, chunks_topic):
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
