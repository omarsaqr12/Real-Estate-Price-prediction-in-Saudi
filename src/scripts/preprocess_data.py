"""Add Arabic text lemmatization to an existing PandA.db Listings table.

This script modifies the input database in place. Back it up before running.
"""
from pathlib import Path
import re
import sqlite3

import pandas as pd
from camel_tools.disambig.mle import MLEDisambiguator

# Refuse to create an empty SQLite file when the expected dataset is absent.
db_path = Path("PandA.db")
if not db_path.is_file():
    raise FileNotFoundError(
        f"Expected existing training database at {db_path.resolve()}. "
        "Obtain a permitted copy and back it up before preprocessing."
    )

# Initialize the MSA disambiguator only after checking that input exists.
mle_msa = MLEDisambiguator.pretrained("calima-msa-r13")

# Connect to the same database filename used by train_model.py.
conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# An existing column is fine; other SQLite failures must be reported.
try:
    cursor.execute("ALTER TABLE Listings ADD COLUMN content_lemmatized TEXT")
except sqlite3.OperationalError as exc:
    if "duplicate column name" not in str(exc).lower():
        raise
    print("Column 'content_lemmatized' already exists. Continuing...")

query = "SELECT rowid, content FROM Listings"
data = pd.read_sql_query(query, conn)
data["content"] = data["content"].fillna("").astype(str)


def safe_content(text):
    text = text.replace('\\', ' ')
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
    return text.strip()


lemmatized_results = []
for i, row in data.iterrows():
    original = row["content"]
    cleaned = safe_content(original)
    print(f"Processing row {i}...")
    if cleaned:
        try:
            analysis = mle_msa.disambiguate(cleaned.split())
            lemmas = [
                token.analyses[0].analysis["lex"] if token.analyses else token.word
                for token in analysis
            ]
            lemmatized = " ".join(lemmas)
        except Exception as exc:
            print(f"Skipping row {i} due to error: {exc}")
            lemmatized = original
    else:
        lemmatized = ""
    lemmatized_results.append((lemmatized, row["rowid"]))

for lemmatized_text, rowid in lemmatized_results:
    cursor.execute(
        "UPDATE Listings SET content_lemmatized = ? WHERE rowid = ?",
        (lemmatized_text, rowid)
    )

conn.commit()
conn.close()
for lemmatized_text, rowid in lemmatized_results[:10]:
    print(f"Row ID: {rowid} | Lemmatized: {lemmatized_text}")
