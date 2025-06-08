# 🧭 Location-Aware Job Matching IR System

This project is a **location-aware job search engine** developed as part of the **SI 650 course at the University of Michigan**. It integrates traditional Information Retrieval (IR) models with neural techniques and supports location-based query understanding to provide highly relevant job search results.

---

## 🔍 Objectives
- Enable natural language job search queries with location awareness  
- Combine BM25 and neural ranking (L2R with cross-encoder features)  
- Provide an intuitive web interface for interactive job exploration  

---

## 🧩 System Architecture

### 📦 Back-End
- **`app.py`**: Flask server for query routing and API integration  
- **`pipeline.py`**: Core data processing, query interpretation, and ranking  
- **`models.py`**: Implements ranking models such as BM25 and L2R  

### 🎨 Front-End
- **`web/home.html`**: Main user interface for query input and result display  
- **PureCSS + Custom JavaScript** for styling and interactivity  

---

## 💡 Features
- **Natural Language Query Parsing**  
  Supports detailed, human-style queries like:  
  _"Masters data science with python machine learning pandas numpy skills around California that provide leadership training programs and employee engagement."_

- **Location Filtering**  
  Detects and prioritizes job postings that match user-specified regions or cities  

- **Hybrid Ranking Model**  
  Combines BM25 with Learning-to-Rank (L2R) and Cross-Encoder features for improved ranking quality  

- **Interactive Search Interface**  
  Simple, responsive job browser with pagination and link-outs to original listings  

---

## ⚙️ Technologies Used
- Python 3.8+  
- Flask  
- BM25 (`rank_bm25` or custom implementation)  
- LightGBM or other L2R tools for learning-to-rank  
- OpenAI GPT (optional, for query/data augmentation)  
- PureCSS + Vanilla JavaScript (UI)  

---

## 🛠️ Setup Instructions

### 🔧 Prerequisites
- Python >= 3.8  
- Flask  
- `requirements.txt` dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- (Optional) GPT API Key for augmentation:
  ```bash
  export OPENAI_API_KEY='your_api_key'
  ```
---

## 📁 Project Structure
  ```bash
├── app.py             # Flask application
├── pipeline.py        # Query processing and ranking logic
├── models.py          # Ranking model implementations
├── web/
│   └── home.html      # Front-end UI
├── requirements.txt   # Python dependencies
└── README.md          # Documentation

  ```
---

## 🚀 Running the Application

1. Start the Flask server by running the `app.py` file with Python.  
   For example: run `python app.py` from your terminal.

2. Open a web browser and navigate to:  
   http://127.0.0.1:8000/

3. Use the search bar to enter a natural language job query.  
   Example:  
   "Masters in data science near California offering mentorship and career growth"

---

## 🧪 Usage Guide

### 🧠 How to Write Queries
You can enter full-sentence queries including:
- **Skills**: e.g., Python, machine learning, SQL
- **Job role or degree**: e.g., Data Scientist, Masters, Software Engineer
- **Location**: e.g., near New York, around California
- **Perks and culture**: e.g., mentorship, flexible hours, remote options

**Examples:**
- "Senior software engineer near New York offering flexible remote work"
- "Data scientist with Python and SQL skills in California with leadership training"

### 🖱️ Interface Behavior
- Results are ranked by relevance
- Use “Next” and “Previous” buttons to browse additional pages
- Click on a job listing to view the original job post

---

## 📌 Query Rules

To get accurate and relevant search results, follow these guidelines:

1. **Use location prepositions** such as:
   - `near`, `in`, `at`, or `around`
   - Example:  
     "Software engineering roles near California with remote options"

2. **Combine multiple filters naturally**:
   - You can include skills, location, role, company culture, and job perks all in one query.
   - Example:  
     "Masters in data science with Python skills in New York offering employee engagement programs"

---

## 🙏 Acknowledgements

- This project was developed for the **SI 650: Information Retrieval** course at the University of Michigan.
- It combines traditional IR models (like BM25) with neural ranking techniques (e.g., L2R and cross-encoders).
- The front-end interface is styled using **PureCSS** and minimal JavaScript for responsiveness.


---


# 🧭 위치 인식 기반 잡 매칭 검색 시스템

이 프로젝트는 **University of Michigan의 SI 650 과목**에서 개발한 **위치 인식 기반 직업 서치 엔진**입니다. 전통적인 정보 검색(IR) 모델과 신경망 기반 모델을 통합하고, 위치 기반 쿼리 이해 기능을 통해 **높은 관련성을 가진 채용 정보를 제공**하는 시스템입니다.

---

## 🔍 프로젝트 목표

- 위치 인식을 반영한 자연어 기반 구직 쿼리 지원  
- BM25와 신경망 기반 랭킹 모델(L2R + Cross-Encoder) 결합  
- 직관적인 웹 인터페이스를 통해 구직 탐색 기능 제공  

---

## 🧩 시스템 아키텍처

### 📦 백엔드

- `app.py`: 쿼리 라우팅과 API 처리를 위한 Flask 서버  
- `pipeline.py`: 핵심 데이터 처리, 쿼리 해석, 랭킹 처리 로직  
- `models.py`: BM25 및 L2R 기반 랭킹 모델 구현  

### 🎨 프론트엔드

- `web/home.html`: 사용자 쿼리 입력 및 결과 표시를 위한 메인 UI  
- PureCSS + 커스텀 JavaScript: 반응형 디자인과 인터랙션 처리  

---

## 💡 주요 기능

- **자연어 쿼리 파싱**  
  사람처럼 길고 복잡한 문장을 이해하여 검색 쿼리로 변환  
  예시:  
  _"캘리포니아 주변에서 리더십 트레이닝과 사내 문화 프로그램을 제공하는 데이터사이언스 석사 졸업자를 위한 머신러닝 관련 잡"_

- **위치 필터링 기능**  
  사용자가 명시한 지역(도시, 주 등)을 자동으로 추출하고 우선순위 적용  

- **하이브리드 랭킹 모델**  
  BM25와 L2R, Cross-Encoder 모델을 결합하여 정교한 결과 제공  

- **인터랙티브 UI**  
  간단하고 직관적인 웹 기반 잡 브라우저 제공 (페이지 이동, 원문 링크 포함)  

---

## ⚙️ 사용 기술

- Python 3.8 이상  
- Flask  
- BM25 (`rank_bm25` 또는 커스텀 구현)  
- LightGBM 또는 기타 L2R 도구  
- (선택) OpenAI GPT API – 쿼리/데이터 증강용  
- PureCSS + Vanilla JavaScript (UI 구성)

---

## 🛠️ 설치 안내

### 🔧 사전 준비

- Python >= 3.8  
- Flask  
- 패키지 설치:
  pip install -r requirements.txt

- (선택) GPT API 키 설정:
  export OPENAI_API_KEY='your_api_key'

---

## 📁 프로젝트 구조

    ├── app.py             # Flask 서버
    ├── pipeline.py        # 쿼리 처리 및 랭킹 로직
    ├── models.py          # 랭킹 모델 구현
    ├── web/
    │   └── home.html      # 프론트엔드 UI
    ├── requirements.txt   # 필요한 라이브러리
    └── README.md          # 프로젝트 문서

---

## 🚀 실행 방법

1. `app.py` 파일을 Python으로 실행하여 Flask 서버를 시작합니다.  
   예시:
   python app.py

2. 웹 브라우저에서 아래 주소로 접속:
   http://127.0.0.1:8000/

3. 자연어로 구직 쿼리를 입력해보세요:  
   예시:  
   _"리더십 트레이닝과 멘토링 기회를 제공하는 캘리포니아 지역의 데이터사이언스 석사 졸업자 대상 채용"_

---

## 🧪 사용 가이드

### 🧠 쿼리 작성 방법

- **기술 스킬**: 예) Python, 머신러닝, SQL  
- **직무/학위**: 예) 데이터 사이언티스트, 석사 졸업, 소프트웨어 엔지니어  
- **지역 조건**: 예) 뉴욕 인근, 캘리포니아 주변  
- **복지/문화**: 예) 유연 근무제, 리더십 훈련, 원격 근무  

**예시 쿼리**:
- "유연한 원격 근무를 지원하는 뉴욕 인근 시니어 소프트웨어 엔지니어"
- "Python과 SQL 가능한 데이터사이언티스트, 캘리포니아 지역, 리더십 교육 제공"

---

### 🖱️ 인터페이스 안내

- 검색 결과는 관련도 기반으로 정렬됨  
- ‘다음’, ‘이전’ 버튼으로 페이지 탐색  
- 결과 항목 클릭 시 원본 채용 공고로 이동  

---

## 📌 쿼리 규칙

정확한 검색 결과를 얻기 위해 다음을 따르세요:

1. **위치 전치사 사용**:
   - `near`, `in`, `at`, `around` 등을 활용
   - 예:  
     "California 주변의 소프트웨어 엔지니어링 직무, 원격 근무 옵션 제공"

2. **자연스럽게 조건 조합**:
   - 기술, 지역, 직무, 기업 문화, 복지를 하나의 쿼리에 포함 가능  
   - 예:  
     "Python 가능한 데이터사이언티스트, 뉴욕 인근, 직원 복지 프로그램 있는 회사"

---

## 🙏 Acknowledgements

- 본 프로젝트는 **미시간대학교 SI 650 정보 검색 과목**의 일환으로 개발되었습니다.  
- 전통적인 IR 모델(BM25)과 신경망 기반 랭킹(L2R, Cross-Encoder)을 결합한 구조입니다.  
- **PureCSS**와 최소한의 JavaScript로 구성된 반응형 인터페이스를 포함합니다.
