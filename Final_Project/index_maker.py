from document_preprocessor import RegexTokenizer, Tokenizer
from indexing import Indexer, IndexType
import sys
import os
import csv
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 인덱스 생성에 필요한 파라미터 설정
index_type = IndexType.BasicInvertedIndex  # 또는 IndexType.PositionalIndex
dataset_path = "./add_company_info.csv"  # 데이터셋 파일 경로


mwe = []
with open('./multi_word_expressions.txt', 'r', encoding='utf-8') as file:
    for line in file:
        mwe.append(line.strip())
        
tokenizer = RegexTokenizer(lowercase=True, multiword_expressions=mwe)

stopwords = set()
with open('./stopwords.txt', 'r', encoding='utf-8') as file:
    for stopword in file:
        stopwords.add(stopword.strip())


minimum_word_frequency = 0  # HW2 지시사항에 따라 50으로 설정
text_key = "description"  # JSON 파일에서 텍스트를 포함하는 키
max_docs = -1  # 모든 문서를 인덱싱하려면 -1, 특정 개수만 원한다면 그 숫자를 입력

# doc_augment_dict 을 doc2query 에서 불러오기
filename = "./add_company_info.csv"
doc_augment_dict = {}

# CSV 파일 읽기
df = pd.read_csv(filename, encoding='utf-8')

# 데이터프레임 내용 확인 (디버깅용)
print(df.head())

# 'docid'와 'company_info' 열을 사용하여 dict 생성
for _, row in df.iterrows():
    doc = int(row['docid'])  # 'docid' 값
    query = row['company_info']  # 'company_info' 값

    if doc not in doc_augment_dict:
        doc_augment_dict[doc] = []
    doc_augment_dict[doc].append(query)

print("doc_augment_dict 생성 완료")
    


# 인덱스 생성

print("Creating title index...")
index = Indexer.create_index(
    index_type=index_type,
    dataset_path=dataset_path,
    document_preprocessor=tokenizer,
    stopwords=stopwords,
    minimum_word_frequency=minimum_word_frequency,
    text_key=text_key,
    max_docs=max_docs,
    doc_augment_dict=doc_augment_dict
)

index.save("augmented_job_description_index")  # 생성된 인덱스 저장

# 생성된 인덱스의 통계 확인
print(index.get_statistics())