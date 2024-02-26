# Motivation

The blow figure shows the performance of ``LLaMA-2-7B`` on the subsets of MMLU dataset. We can find that ``LLaMA-2-7B`` is weak in STEM area compared with the other two. In the Machine Learning domain,``LLaMA-2-7B`` only achieves less than 50% accuracy. This project aims to probe whether utilizing RAG could imporve ``LLaMA-2-7B``'s model performance in Machine Learning QA task.

![MMLU](./pics/compare_domains_mmlu_llama-2-7b.png)

# Repo Overview
```
Machine Learning RAG/
│
├── data/                                 
│   └── wiki_ml_mediawiki_cleaned.parquet        
│
├── dataset/                              
│   ├── extract_wikidata.ipynb            
│   └── mmlu_data.py                      
│
├── interface/                            
│   ├── get_response.py                   
│   ├── inference.py                      
│   └── UI.py                             
│
├── model/                                
│   ├── embedding_model.py                
│   └── inference_model.py               
│
├── RAG/                                  
│   ├── chunk_data.py                     
│   ├── index_data.py                     
│   ├── main.py                           
│   └── utils.py                          
│
└── wikiextractor/   #https://github.com/attardi/wikiextractor

```
# Dataset

## Wiki Machine Learning Corpus

We scrape the plain text from Wikipedia with [WikiAPI](https://github.com/lehinevych/MediaWikiAPI). ``Statistics``, ``Artificial intelligence``, ``Computational mathematics``, ``Numerical analysis``, ``Applied mathematics``, ``Probability`` supercategories that include Machine Learning knowledges are selected as text resources. For each category, recursively collect the root pages.
Extract the page contents of the collected wiki URLs Wikipedia-API (~7k pages).

![wiki_super](./pics/wiki_super.png)

## MMLU Machine Learning QA subset

We utilize the test set in [MMLU](https://huggingface.co/datasets/lukaemon/mmlu/viewer/machine_learning/train) as our RAG system test set. The train set in it is utilized to as few-shot samples included in the prompt.

# Pipeline 
![MMLU](./pics/pipeline.png)



1. **Data Collection**: Scrape Machine Learning related text with [Wiki Pages](https://en.wikipedia.org/wiki/Category:Artificial_intelligence). Then the noisy sections in the text data are remove through [Wikiextractor Repo](https://github.com/attardi/wikiextractor).
2. **Chunking**: The cleand text data is chunked through sliding window with 200 words as window size and 50 as sliding step size.
3. **Embed Text**: The text is embedded with BGE-Large model which is selected through [MTEB LeaderBoard](https://huggingface.co/spaces/mteb/leaderboard).
4. **Index Embedding**: The embeddings are indexed into [Pinecone](https://www.pinecone.io/). The retriver utilizes cosine similarity to retrieve relavant embeddings from the database.
5. **Prompting**: The query will be embedded with the same encoder. Then the retrieved text will be added into the prompt.
6. **Inference**: ``LLaMA-2-7B`` model is utilized to generate results. Due to the autogressive nature, the generated text is post-processed and only the first answer is extracted as the final decision.

## Results
As shown in the results, the generated context information successfully support the machine learning question.
![MMLU](./pics/results.png) 
