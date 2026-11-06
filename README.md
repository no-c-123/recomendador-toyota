# Toyota KMS — Document Recommender (TF-IDF)

**Author:** Héctor Emiliano Leal Prieto (AL03010122)
**Course:** Sistemas y tecnologías de administración del conocimiento — Tecmilenio

## Objective

Recommend the **Top-5 most similar documents** from the knowledge repository of a Toyota plant, so that an operator, engineer or support agent who opens one document (or types a question) immediately finds the related policies, incidents and training. The system uses **Path A: TF-IDF + cosine similarity**.

## Folder contents

| File / folder | Description |
|---|---|
| `recommender.py` | Recommender system (TF-IDF, cosine similarity, CLI and interactive menu) |
| `build_corpus.py` | Script that generates `corpus.csv` |
| `corpus.csv` | 40 simulated Toyota documents — columns `id`, `title`, `text`, `category` |
| `evidencias/` | Console captures (`.png`), raw outputs (`.txt`) and comments of the three tests |
| `Reporte_Recomendador_HectorLeal.pdf` | Short report: approach, areas of opportunity and video link |

**Corpus categories:** Policy/Process (11), Support/Incidents (11), Quality (7), Training (7), Supply Chain (4).

## Dependencies

- Python 3.10+
- pandas
- scikit-learn

```bash
pip install pandas scikit-learn
```

## Execution steps

1. Open a terminal in this folder.
2. (Optional) Regenerate the dataset: `python build_corpus.py`
3. Run the recommender in one of these modes:

```bash
python recommender.py                    # interactive menu
python recommender.py --id DOC001        # Option 1: seed document ID
python recommender.py --text "question"  # Option 2: free text query
python recommender.py --demo             # runs the three evidence tests
```

## How it works

1. **Cleaning:** lowercase, removal of special characters and English stopwords.
2. **Vectorization:** `TfidfVectorizer` (unigrams + bigrams, sublinear TF) over title + text.
3. **Similarity:** cosine similarity between every pair of documents (or between the query and every document).
4. **Ranking:** returns the Top-5 by score; the seed document is excluded when searching by ID.
5. **Validation:** unknown IDs and empty queries return a clear error instead of crashing.

## Usage example

```text
$ python recommender.py --id DOC011
Input (seed ID): DOC011 | Support/Incidents | Ticket INC-1042: MES terminal frozen on assembly line 2

#  ID      Score   Category           Title
------------------------------------------------------------------------
1  DOC032  0.286   Training           Training course: MES terminal basics ...
2  DOC012  0.156   Support/Incidents  Ticket INC-1057: Torque tool not send...
3  DOC019  0.118   Support/Incidents  Helpdesk FAQ: MES and shop floor systems
4  DOC027  0.041   Quality            Quality report: Water leak root cause
5  DOC017  0.040   Support/Incidents  Incident report: Hybrid battery handl...
```

## Note on the data

The documents are **simulated** internal documents (policies, tickets, meeting minutes, FAQs, training) written for this activity, using Toyota as the case study. They are not real Toyota documents.

## Organizational reflection video

https://youtu.be/f-_jTOZWBf8
