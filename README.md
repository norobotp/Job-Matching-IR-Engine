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
