import pandas as pd
from tqdm import tqdm


def sliding_window(wiki_data, chunck_size = 20, step_size=10):
    chunks = []
    chunks_topic =  []
    
    for i in tqdm(range(len(wiki_data))):
        topic = wiki_data[i][0]
        text = wiki_data[i][1]
        words = text.split()
    
        for i in range(0, len(words) - chunck_size + 1, step_size):
            chunks.append(" ".join(words[i:i + chunck_size]))
            chunks_topic.append(topic)
    return chunks, chunks_topic
