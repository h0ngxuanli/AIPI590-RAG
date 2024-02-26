from datasets import load_dataset

def load_data()

dataset = load_dataset("lukaemon/mmlu", "machine_learning", split="test")
few_shots = load_dataset("lukaemon/mmlu", "machine_learning", split="train")
test_df = pd.DataFrame(dataset)
test_df.iloc[[0], :]