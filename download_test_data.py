import pandas as pd
from datasets import load_dataset


def download_bitext_dataset():
    print("Connecting to Hugging Face...")

    # Load the specific Bitext dataset mentioned in the project appendix
    # We load the "train" split which contains the conversational intents
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")

    print(f"Successfully downloaded {len(dataset)} customer support records.")

    # Convert the Hugging Face dataset to a Pandas DataFrame for easier handling
    df = dataset.to_pandas()

    # Optional: Filter down to just the columns you need for evaluation
    # 'instruction' is the customer's message, 'response' is the expected AI answer
    if 'instruction' in df.columns and 'response' in df.columns:
        df = df[['instruction', 'intent', 'category', 'response']]

    # Save it locally so you don't have to re-download it every time you run your evals
    output_filename = "evaluation_happy_paths.csv"
    df.to_csv(output_filename, index=False)

    print(f"Dataset saved locally as {output_filename}")
    print("Ready for your Observability & Eval dashboard!")


if __name__ == "__main__":
    download_bitext_dataset()