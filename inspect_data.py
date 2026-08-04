from datasets import load_dataset
from huggingface_hub import HfApi
import sys

def search_hf(query):
    api = HfApi()
    models = api.list_datasets(search=query, limit=5)
    return [m.id for m in models]

datasets_to_load = [
    "deepset/prompt-injections",
    "jackhhao/jailbreak-classification",
    "qualifire/prompt-injections-benchmark",
    "rubend18/ChatGPT-Jailbreak-Prompts",
    "HuggingFaceH4/no_robots"
]

print("Searching for safe-guard-prompt-injection...")
matches = search_hf("safe-guard-prompt-injection")
print(f"Matches found: {matches}")

if matches:
    datasets_to_load.append(matches[0])
else:
    print("Warning: Could not find safe-guard-prompt-injection")

print("\n--- Inspecting Datasets ---")
for ds_name in datasets_to_load:
    try:
        print(f"\n[{ds_name}]")
        ds = load_dataset(ds_name)
        print("Splits:", list(ds.keys()))
        # take the first split (usually 'train')
        split_name = list(ds.keys())[0]
        print("Features:", ds[split_name].features)
        print("Column names:", ds[split_name].column_names)
        print("Sample row:", ds[split_name][0])
    except Exception as e:
        print(f"Error loading {ds_name}: {e}")
