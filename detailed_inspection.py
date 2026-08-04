import pandas as pd
from datasets import load_dataset
import os

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

datasets_to_load = {
    "deepset": "deepset/prompt-injections",
    "jackhhao": "jackhhao/jailbreak-classification",
    "rubend18": "rubend18/ChatGPT-Jailbreak-Prompts",
    "no_robots": "HuggingFaceH4/no_robots",
    "safe-guard": "xTRam1/safe-guard-prompt-injection"
    # qualifire is skipped due to being gated
}

print("="*60)
print("DATA LOADING & INSPECTION REPORT")
print("="*60)

for name, repo in datasets_to_load.items():
    print(f"\n[{name}] -> {repo}")
    try:
        ds = load_dataset(repo)
        split_name = list(ds.keys())[0]
        df = ds[split_name].to_pandas()
        
        print(f"Schema / Columns: {list(df.columns)}")
        print(f"Total rows in '{split_name}': {len(df)}")
        
        print(f"Nulls check:\n{df.isnull().sum().to_string()}")
        
        # Check raw class balance if there's an obvious label column
        label_col = None
        for col in ['label', 'type', 'category']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            print(f"Raw class balance ({label_col}):\n{df[label_col].value_counts().to_string()}")
        else:
            print("No obvious label column found for class balance.")
            
        print("Example row:")
        example_dict = df.iloc[0].to_dict()
        # Truncate long strings for printing
        for k, v in example_dict.items():
            if isinstance(v, str) and len(v) > 100:
                example_dict[k] = v[:100] + "..."
        print(example_dict)
        
    except Exception as e:
        print(f"Failed to load {repo}: {e}")
    print("-" * 60)
