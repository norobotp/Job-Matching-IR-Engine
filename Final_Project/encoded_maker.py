from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_raw_text_dict(csv_file_path: str) -> dict[int, str]:
    """
    문서의 처음 500단어만 포함하는 딕셔너리를 생성합니다.
    """
    # CSV 파일을 DataFrame으로 읽어오기
    df = pd.read_csv(csv_file_path)

    # doc_id와 description의 처음 500자까지 추출하여 딕셔너리 생성
    documents = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading raw text dictionary"):
        doc_id = int(row['docid'])
        description = row['description']
        documents[doc_id] = description

    return documents

# Step 1: Define your documents (example list of document texts)
documents = load_raw_text_dict('updated_jobs_docid.csv')

# Step 2: Load the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 3: Encode the documents
encoded_docs = model.encode(list(documents.values()), convert_to_tensor=False, show_progress_bar=True)

# Step 4: Convert to numpy array (already numpy if `convert_to_tensor=False`)
encoded_docs = np.array(encoded_docs)

# Step 5: Save the encoded documents in .npy format
np.save('encoded_jobs.npy', encoded_docs)
