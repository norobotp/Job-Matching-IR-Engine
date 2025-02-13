# **Location-Aware Job Matching IR System**

This project is a location-aware job search engine built for the SI 650 course. It integrates traditional IR models (e.g., BM25) with neural approaches and includes a location filtering feature to provide highly relevant job search results.

---

## **Table of Contents**
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Running the Application](#running-the-application)
- [System Components](#system-components)
- [Usage](#usage)
- [Query Rules](#query-rules)
- [Acknowledgements](#acknowledgements)

---

## **Features**
- **Natural Language Query Processing**:  
  Supports complex queries like:  
  _"Masters data science with python machine learning pandas numpy skills around California that provide leadership training programs, employee engagement initiatives, and a supportive team culture with opportunities for growth."_  
- **Location Filtering**:  
  Matches job locations to user-specified queries.  
- **Hybrid Ranking**:  
  Implements an L2R (Learning-to-Rank) ranker with a BM25 base ranker and cross-encoder features for better relevance.  
- **Interactive Web Interface**:  
  A user-friendly search interface for browsing and navigating job listings.

---

## **Prerequisites**
Before running the system, ensure the following are installed:
1. **Python** (version >= 3.8)
2. **Flask** (for serving the web application)
3. **Dependencies** listed in `requirements.txt` (run `pip install -r requirements.txt`)
4. **HTML/CSS** for front-end rendering.
5. **Access to GPT API** (optional, for additional data augmentation).

---

## **Setup Instructions**
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the following project structure:
   ```
   ├── app.py             # Main Flask application
   ├── pipeline.py        # Data processing and ranking pipeline
   ├── models.py          # Implementation of ranking models
   ├── web/
       └── home.html      # Front-end template
   ├── requirements.txt   # Python dependencies
   └── README.md          # Documentation
   ```

4. **(Optional)** Set up API keys for GPT or other external tools in your environment variables:
   ```bash
   export OPENAI_API_KEY='your_api_key'
   ```

---

## **Running the Application**
1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

3. Use the search bar to input a job query (e.g., _"Masters in data science near California"_).

---

## **System Components**
### **1. Back-End**
- **`app.py`**: Handles routing, query processing, and API calls.
- **`pipeline.py`**: Processes user queries, retrieves job postings, and applies ranking models.
- **`models.py`**: Contains ranking algorithms, including BM25 and L2R implementations.

### **2. Front-End**
- **`./web/home.html`**: Provides a user interface for the search engine, including input fields and result displays.
- **CSS/JavaScript**: Enhances interactivity with responsive design and pagination.

---

## **Usage**
1. **Enter Query**: Input a natural language query in the search bar, specifying job criteria such as location, skills, and work culture.  
   Example:  
   _"Senior software engineer near New York offering flexible remote work."_  
2. **View Results**: Job listings are ranked by relevance and displayed interactively.  
3. **Navigate Pages**: Use "Next" and "Previous" buttons to browse results.  
4. **Click Details**: Each job listing includes a link to the original posting for more information.

---

## **Query Rules**
To ensure accurate results, follow these rules when constructing your query:
1. Use **prepositions for location**:
   - Write **near, at, around, or in** before the location name.  
   - Example:  
     _"Software engineering roles **near California** with remote options."_  
2. Combine multiple criteria in natural language:
   - Skills, location, company preferences, and job perks can all be included.  
   - Example:  
     _"Masters in data science with python skills **at New York** providing mentorship programs."_  

---

## **Acknowledgements**
- Developed as part of the SI 650 course project at the University of Michigan.  
- Inspired by traditional IR methods and enhanced with neural ranking techniques.  
- Front-end designed with **PureCSS** and custom styles.  

---

