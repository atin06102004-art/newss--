# 📰 Fake News Detector

A machine learning web app that classifies news claims/headlines using a TF-IDF + classic ML baseline, then **verifies the claim against live web evidence** to correct the model when it's wrong.

🔗 **Live Demo:** *add your Streamlit Cloud link here after deploying*

## How it works

```
News input
    ↓
1. ML Model (TF-IDF + Logistic Regression / Random Forest) → initial prediction
    ↓
2. Live web search for the same claim (Tavily)
    ↓
3. Evidence/verification layer compares search results with the claim
    ↓
4. Final verdict: 🟢 VERIFIED REAL   |   🔴 LIKELY FAKE   |   🟡 UNVERIFIED
```

The ML model can be wrong — especially on recent events it never saw during training. Live evidence is used to confirm or override it. Not finding an article does **not** prove a claim is fake, so that case is marked **UNVERIFIED** rather than **FAKE**.

## Features

- **Text preprocessing pipeline** — cleaning, stopword removal, lemmatization (NLTK)
- **TF-IDF vectorization** (unigrams + bigrams)
- **Model comparison** — Logistic Regression, Naive Bayes, and Random Forest trained and benchmarked automatically; best model auto-selected
- **Live web verification** — searches the web for the same claim via **Tavily** and cross-checks results
- **3-way verdict** — VERIFIED REAL / LIKELY FAKE / UNVERIFIED, instead of a binary Real/Fake
- **Explainability** — LIME highlights which words drove the ML prediction
- **Interactive Streamlit UI**

## Tech Stack

`Python` `scikit-learn` `Streamlit` `NLTK` `LIME` `Tavily` `Pandas` `NumPy`

## Project Structure

```
fake-news/
├── data/                  # place Fake.csv and True.csv here
├── models/                # saved model + vectorizer (generated after training)
├── utils.py               # text cleaning/preprocessing
├── verify.py               # live web search + evidence/verification layer (Tavily)
├── train_model.py         # trains and compares models, saves the best one
├── app.py                 # Streamlit app
├── requirements.txt
├── .env.example            # template for your Tavily API key
└── README.md
```

## Setup

1. **Clone the repo**

   ```
   git clone https://github.com/atin06102004-art/fake-news.git
   cd fake-news
   ```

2. **Install dependencies**

   ```
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   Get the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` and `True.csv` inside the `data/` folder.

4. **Get a Tavily API key**
   Sign up for free at [tavily.com](https://tavily.com), then copy `.env.example` to `.env` and paste your key in:

   ```
   TAVILY_API_KEY=your_key_here
   ```

   On Streamlit Cloud, set `TAVILY_API_KEY` under **Settings → Secrets** instead.

5. **Train the model**

   ```
   python train_model.py
   ```

   This prints accuracy/precision/recall for all three models and saves the best one to `models/`.

6. **Run the app**

   ```
   streamlit run app.py
   ```

## Model Performance

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~0.98    |
| Naive Bayes         | ~0.94    |
| Random Forest       | ~0.99    |

*(Fill in your actual numbers after training — they'll print in the terminal.)*

## Verdict Logic

| Condition                                                        | Verdict           |
| ------------------------------------------------------------------ | ------------------ |
| 2+ relevant sources found supporting the claim                     | 🟢 VERIFIED REAL   |
| Only 1 loosely related source                                      | 🟡 UNVERIFIED      |
| No relevant sources found **and** ML says FAKE with ≥70% confidence | 🔴 LIKELY FAKE     |
| No relevant sources found, otherwise                                | 🟡 UNVERIFIED      |

## Future Improvements

- Swap the keyword-overlap relevance check in `verify.py` for embedding-based semantic similarity
- Fine-tune a BERT model and compare against the classical ML baseline
- Deploy with metadata features (source, author, publish date)

## Author

Atin Choudhary — [GitHub](https://github.com/atin06102004-art) · [Email](mailto:atin06choudhary@gmail.com)
