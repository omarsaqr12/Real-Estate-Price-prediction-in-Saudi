import sqlite3
import pandas as pd
import re
from camel_tools.disambig.mle import MLEDisambiguator

# Initialize the MSA disambiguator
mle_msa = MLEDisambiguator.pretrained("calima-msa-r13")

# Connect to the SQLite database
db_path = "database.db"  # Replace with your actual database file name
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Try adding the new column if it doesn't already exist
try:
    cursor.execute("ALTER TABLE Listings ADD COLUMN content_lemmatized TEXT")
except sqlite3.OperationalError:
    print("Column 'content_lemmatized' already exists. Continuing...")

# Read content and rowid from the database
query = "SELECT rowid, content FROM Listings"
data = pd.read_sql_query(query, conn)

# Fill NaN values with empty strings
data["content"] = data["content"].fillna("").astype(str)

# Helper function to clean suspicious characters
def safe_content(text):
    text = text.replace('\\', ' ')  # Remove problematic backslashes
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # Keep Arabic, digits, and whitespace
    return text.strip()

# Lemmatize or fallback to original if error occurs
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
        except Exception as e:
            print(f"Skipping row {i} due to error: {e}")
            lemmatized = original  # Fallback to original
    else:
        lemmatized = ""  # Empty content remains empty

    lemmatized_results.append((lemmatized, row["rowid"]))

# Bulk update in database
for lemmatized_text, rowid in lemmatized_results:
    cursor.execute(
        "UPDATE Listings SET content_lemmatized = ? WHERE rowid = ?",
        (lemmatized_text, rowid)
    )

# Commit and close connection
conn.commit()
conn.close()

# Optionally print summary
for lemmatized_text, rowid in lemmatized_results[:10]:  # Show first 10
    print(f"Row ID: {rowid} | Lemmatized: {lemmatized_text}")
