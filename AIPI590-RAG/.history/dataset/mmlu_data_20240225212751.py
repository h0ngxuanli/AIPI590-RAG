from datasets import load_dataset
import pandas as pd 


def load_data():
    """
    The function loads two splits of a dataset named "lukaemon/mmlu" from an online repository. 

    :return: A tuple containing two Pandas DataFrames: 
             - The first DataFrame corresponds to the 'test' split of the dataset.
             - The second DataFrame corresponds to the 'train' split of the dataset.
    """
    dataset = load_dataset("lukaemon/mmlu", "machine_learning", split="test")
    few_shots = load_dataset("lukaemon/mmlu", "machine_learning", split="train")
    test_df = pd.DataFrame(dataset)
    few_shots_df = pd.DataFrame(few_shots)
    return test_df, few_shots_df
    