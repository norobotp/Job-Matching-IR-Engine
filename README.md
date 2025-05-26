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
