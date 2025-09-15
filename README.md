# Content‑Based Music Recommendation

Aman Sharma
_Project type:_ Coursework project and research notebook  
_Goal:_ Build a content‑based music recommender that classifies songs into topical themes and recommends new tracks aligned with user interests.

---

## Overview

This repository documents a complete pipeline for a content‑based recommender on a curated dataset of **1,500 songs** labelled into **5 topics**: `dark`, `emotion`, `lifestyle`, `personal`, `sadness`. Each record contains `artist_name`, `track_name`, `release_date`, `genre`, `lyrics`, and `topic`.

The work proceeds in three parts:

1. **Topic Classification** - transform text to features, try classical and transformer models, and compare metrics.  
2. **Recommendation Methods** — build user topic‑profiles with TF‑IDF and cosine similarity, then rank new songs.  
3. **User Evaluation** — simulate multi‑week usage and report ranking metrics such as Hit Rate@N, MAP, Recall, and Diversity.

The project is implemented in a single notebook: `Music-Rec.ipynb`. This README extracts, organises, and explains the methods and results in a reproducible, engineering‑ready format.

---

## Dataset

- Size: **1,500 songs**
- Labels: **5 topics** (`dark`, `emotion`, `lifestyle`, `personal`, `sadness`)
- Fields: `artist_name`, `track_name`, `release_date`, `genre`, `lyrics`, `topic`
- Notes:
  - The dataset is **class‑imbalanced** (more `dark`/`sadness`, fewer `emotion`), so macro‑averaged metrics are used to reflect performance across classes.
  - Free‑form **lyrics** are the main signal; auxiliary metadata (artist, genre, date) are concatenated for several baselines.

---

## Text Preprocessing

Two progressively stronger cleaning functions are used.

1. **Tutorial Preprocessing**
   - Lower‑casing
   - Remove punctuation via regex
   - `word_tokenize`
   - Remove English stopwords
   - **Porter stemming**
   - Produces `cleaned_doc`

2. **Final Preprocessing**
   - Keep apostrophes for contractions
   - Lower‑casing
   - `word_tokenize`
   - Remove English stopwords
   - **Snowball stemming** (slightly less aggressive than Porter on some corpora)
   - Produces `cleaned_lyrics`

> _Why stemming?_ Stemming reduces inflectional variants (e.g., “running”→“run”), which compresses the vocabulary and improves generalisation for bag‑of‑words models.

The following composite fields are constructed for experiments:

- **`document`** = artist + track + release_date + genre + lyrics  
- **`cfd`** (“cleaned final document”) = artist + track + release_date + genre + `cleaned_lyrics`

---

## Part 1 — Topic Classification

### Feature Extraction
- **CountVectorizer** for Naive Bayes baselines.
- **TF‑IDF** for SVM and for recommendation profiles.
- A **vocabulary sweep** over `top‑N` terms is run for NB baselines: `N ∈ {500, 1000, 2000, …, 8000, All}`.

### Models
- **Bernoulli Naive Bayes (BNB)**
- **Multinomial Naive Bayes (MNB)**
- **Linear Support Vector Machine (LinearSVC)**
- **BERT** (`bert-base-uncased`) fine‑tuning with Hugging Face `Trainer`
- **DistilBERT** (`distilbert-base-uncased`) fine‑tuning with `Trainer`

### Validation Protocol
- **StratifiedKFold (k=5, shuffle=True, random_state=42)**
- Report **Accuracy**, **Precision (macro)**, **Recall (macro)**, **F1 (macro)**
- For Transformers:
  - Tokeniser: model‑specific (`BertTokenizer`, `DistilBertTokenizerFast`)
  - Loss: cross‑entropy (handled by `Trainer`)
  - Batch size: **8**
  - Epochs: **10**
  - `load_best_model_at_end=True`, `metric_for_best_model="accuracy"`
  - Metrics computed with `evaluate`

### Key Results (cross‑validation means ± std)

| Model        | Accuracy | Precision (Macro) | Recall (Macro) | F1 (Macro) |
|--------------|---------:|------------------:|---------------:|-----------:|
| **BNB**      | 0.5327 ± 0.0173 | 0.4110 ± 0.0325 | 0.3888 ± 0.0120 | 0.3524 ± 0.0116 |
| **MNB**      | 0.7773 ± 0.0129 | 0.7458 ± 0.0380 | 0.6883 ± 0.0129 | 0.7042 ± 0.0196 |
| **LinearSVC**| **0.8407 ± 0.0165** | **0.8297 ± 0.0197** | **0.8114 ± 0.0364** | **0.8188 ± 0.0282** |
| **BERT**     | ~0.8433 (from notebook chart) | ~0.8148 | ~0.8409 | ~0.8254 |
| **DistilBERT** | ~0.8400 | ~0.8197 | ~0.8121 | ~0.8157 |

**Observations**  
- **LinearSVC** is a very strong classical baseline, competitive with fine‑tuned transformers on this dataset.  
- **MNB** benefits significantly from larger vocabularies; performance plateaus around **top‑N ≈ 1,000–2,000**, which is adopted for SVM.  
- Given data scale and label granularity, classical models are efficient and accurate. Transformers reach similar performance with more compute.

> _note:_ When dataset scale is modest and labels capture coarse topical themes, LinearSVC + TF‑IDF often provides a superior time‑to‑accuracy ratio. Transformers become more advantageous with more data and richer objectives.

---

## Part 2 — Recommendation Methods

The recommender uses a **content‑based, topic‑aware** approach:

1. **Train a topic classifier** on Weeks 1–3 (first 750 songs) using TF‑IDF + LinearSVC.  
2. **Predict topics** for the same window to create per‑topic corpora.  
3. Build **per‑topic TF‑IDF vectorisers** and **user topic‑profiles** by aggregating lyrics that contain user‑supplied keywords per topic.  
4. For a new song in Week 4, rank by **cosine similarity** between the song TF‑IDF vector and the user’s profile vector for the song’s predicted topic.  
5. Optionally restrict each topic vector to the **top‑M** TF‑IDF terms to control profile sparsity (tested `M ∈ {20, 50, all}`).

### Metrics

- **Hit Rate@5**: whether any relevant song is present among top 5.  
- **Mean Average Precision (MAP)**: order‑sensitive precision averaging per query.  
- **Recall@5** per topic.  
- **Diversity**: average pairwise dissimilarity among recommended items, computed as `1 − cosine_similarity`.

### Hit Rate@5 (from notebook)

| User   | M=20 | M=50 | M=all |
|:-------|-----:|-----:|------:|
| User 1 | 1.0  | 1.0  | 1.0   |
| User 2 | 0.2  | 0.2  | 0.4   |
| User 3 | 0.8  | 0.8  | 0.8   |

**Interpretation**  
- **User 1** is well‑captured; profile sparsity does not hurt.  
- **User 2** benefits from a richer profile (**M=all**), suggesting broader or diffuse interests.  
- **User 3** is robust to M.  
- In practice, one can adapt **M per user** using validation on historical clicks, or learn topic‑specific weights.

> _Cold‑start note:_ The approach mitigates item cold‑start for new songs as long as lyrics are available, since ranking is based on content rather than interactions.

---

## Part 3 — Simulated User Evaluation

A four‑week simulation is run for a single subject (“User 1”):

1. **Weeks 1–3** form the training window. The classifier predicts topics; per‑topic TF‑IDF models are fit; the user profile is built by aggregating the lyrics of liked items per topic.  
2. **Week 4** recommendations are generated by cosine similarity within the matching topic.  
3. Metrics reported: **Precision@5**, **Recall@5**, **Hit Rate@5**, **MAP@5**, **Diversity**.

> The notebook prints a compact results table for “User 1 (subject)”. Values will vary with keyword choices and random seeds. The design cleanly separates feature learning (TF‑IDF), preference modelling (profiles), and ranking (cosine), which allows targeted ablations.

---

## What Worked and Why

- **LinearSVC + TF‑IDF** strikes the best balance of **accuracy, speed, and stability** given data scale.  
- **Per‑topic profiles** avoid mixing signals across topics and reduce drift.  
- **Top‑M term capping** is a simple control over profile sharpness, useful for users with focused interests.

## Limitations

- **Label imbalance** affects macro metrics, especially for minority topics such as `emotion`.  
- **Keyword‑driven profiles** require manual curation. For production, learn profiles from clicks and skips.  
- **Transformers** were run with limited epochs and batch size; stronger schedules and regularisation can help.

---

## Extensions and Next Steps

- **Better text features**: character n‑grams, subword models, or domain‑specific tokenisation for lyrics.  
- **Neural recommenders**: dual encoders that learn user and item embeddings end‑to‑end.  
- **Hybrid signals**: incorporate audio descriptors (tempo, key, MFCCs) and basic metadata.  
- **Calibration and fairness**: ensure balanced exposure across topics and artists.  
- **Online learning**: update user profiles continuously from feedback.

---

## Reproducing the Experiments

1. Run **Part 1** cells to reproduce the classification benchmarks.  
   - For NB sweeps, ensure the vocabulary grid (`vocab_sizes`) is set.  
   - For Transformers, check that PyTorch and `transformers` are installed and a GPU is available.

2. Run **Part 2** to train per‑topic TF‑IDF models and generate recommendations with different **M** values.

3. Run **Part 3** to simulate the week‑by‑week evaluation and compute user‑level metrics.

Set seeds wherever available (`random_state=42`) for comparability.


---

## 🔗 Related Work and Inspiration

- TF‑IDF and LinearSVC for text classification are strong baselines in many domains with short‑to‑medium length documents.  
- Transformer fine‑tuning often surpasses classical models with more data and careful schedules.  
- Content‑based recommenders remain robust for cold‑start and explainability since they score items directly from features.

---

## License

This project is for academic purposes. If you plan to reuse the code for research or production, please include attribution to the author and verify licensing of any datasets used.

---

## Acknowledgements

- Hugging Face `transformers` and `evaluate`  
- Scikit‑learn for classical text classification  
- NLTK for tokenisation and stemming

