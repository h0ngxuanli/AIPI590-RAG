import pandas as pd
from tqdm import tqdm


def sliding_window(wiki_data, chunck_size = 20, step_size=10):
    
    """
    Dividing text into chunks using a sliding window approach.

    :param chunck_size: The number of words in each text chunk (default is 20).
    :param step_size: The step size for the sliding window to move over the text (default is 10).

    :return: A tuple of two lists: 
             - The first list contains chunks of text.
             - The second list contains the corresponding topics for each chunk.
    """
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
