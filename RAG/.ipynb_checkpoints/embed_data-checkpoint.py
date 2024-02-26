from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np




class Encoder:
    def __init__(self, checkpoint, MAX_SEQ_LEN, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.sentences= None
        self.MAX_SEQ_LEN = MAX_SEQ_LEN

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=self.MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        self.sentences= sentences
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader
        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings