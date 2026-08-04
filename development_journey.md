# Prompt Injection Predictor: Development Journey

*This document serves as a detailed log of the development phases, architectural decisions, and data transformations for this project. It is intended to help explain the methodology and engineering process to stakeholders/professors.*

---

## Phase 1: Project Setup and Data Harmonization

### The Goal
The objective was to bootstrap the machine learning pipeline and construct a robust, unified dataset for prompt injection detection. Since the originally planned IEEE dataset required a paid subscription, we shifted to a more defensible approach: building a consolidated dataset from multiple open-source Hugging Face repositories.

### What We Did
1. **Infrastructure**: We established the project structure, creating directories for data (`data/`), exploratory data analysis plots (`outputs/eda/`), models (`outputs/models/`), and evaluation metrics (`outputs/eval/`). We also created a strict `requirements.txt` ensuring reproducibility (pinning `xgboost==1.7.6` to avoid known compatibility bugs with `shap`).
2. **Data Acquisition**: We programmatically fetched five disparate datasets directly from Hugging Face using the `datasets` library. 
   - `deepset/prompt-injections`
   - `jackhhao/jailbreak-classification`
   - `rubend18/ChatGPT-Jailbreak-Prompts`
   - `HuggingFaceH4/no_robots`
   - `xTRam1/safe-guard-prompt-injection`
   *(Note: `qualifire/prompt-injections-benchmark` was skipped because it is gated and requires an authentication token, but the script is built to gracefully handle its absence).*
3. **Harmonization**: We mapped the radically different column names (`text`, `prompt`, `label`, `type`) from each source into a single unified schema: 
   - `text`: The prompt string.
   - `label`: Binary indicator (0 for benign, 1 for malicious).
   - `attack_type`: Categorization of the attack (if known).
   - `source`: The origin dataset for traceability.
4. **Data Cleaning**: We dropped duplicates and NaN values, resulting in a finalized `merged_corpus.csv` containing over 9,000 pristine samples.

### Why We Did It
Combining heterogeneous data sources prevents our models from overfitting to the quirks of a single dataset (a common pitfall in NLP security research). Creating a unified schema is a critical first step in data engineering, allowing downstream ML algorithms to ingest the data seamlessly.

---

## Phase 2: Pipeline Generation and Mock Behavioral Data (In Progress)

### The Goal
To build the end-to-end Jupyter Notebook (`project.ipynb`) that handles EDA, Preprocessing, Modeling (Stages 1-3), and Explainability.

### What We Did
1. **Notebook Generation**: We wrote a Python script to programmatically construct the Jupyter notebook. This notebook encapsulates the entire 12-stage plan.
2. **Mock Behavioral Red-teaming**: A core novelty of this project is predicting actual *bypass likelihood* (Stage 3). To achieve this, the pipeline is designed to query the Anthropic API (Claude) with malicious prompts and programmatically judge if Claude refuses or complies. Because we don't have an API key configured yet, we implemented a graceful fallback that generates a "mock" behavioral dataset (`behavioral_sample.csv`). This ensures the ML pipeline can be tested end-to-end immediately.
3. **Model Training**: 
   - **Stage 1 (Binary)**: XGBoost model to detect any injection.
   - **Stage 2 (Multi-class)**: XGBoost to classify the type of attack.
   - **Stage 3 (Severity)**: XGBoost ordinal regression on the behavioral sample to predict bypass success.
4. **SHAP Explainability**: Implemented SHAP (SHapley Additive exPlanations) to identify exactly which words or n-grams are most responsible for driving a high bypass likelihood.

### Why We Did It
By using mock data for Stage 3, we decouple the engineering of the pipeline from external API dependencies. This allows us to guarantee that the machine learning code (feature extraction, training, evaluation, and SHAP plots) works perfectly. Once an API key is provided, the exact same pipeline will seamlessly slot in real behavioral data.

*(Will be updated as development continues...)*
