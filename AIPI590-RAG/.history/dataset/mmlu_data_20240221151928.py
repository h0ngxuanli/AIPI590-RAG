from datasets import load_dataset
import pandas as pd 


def load_data():
    dataset = load_dataset("lukaemon/mmlu", "machine_learning", split="test")
    few_shots = load_dataset("lukaemon/mmlu", "machine_learning", split="train")
    test_df = pd.DataFrame(dataset)
    few_shots_df = pd.DataFrame(few_shots)
    return test_df, few_shots_df
    