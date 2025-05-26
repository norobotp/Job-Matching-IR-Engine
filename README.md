Location-Aware Job Matching IR System
This project is a location-aware job search engine built for the SI 650 course. It integrates traditional IR models (e.g., BM25) with neural approaches and includes a location filtering feature to provide highly relevant job search results.

Table of Contents
Features
Prerequisites
Setup Instructions
Running the Application
System Components
Usage
Query Rules
Acknowledgements
Features
Natural Language Query Processing:
Supports complex queries like:
"Masters data science with python machine learning pandas numpy skills around California that provide leadership training programs, employee engagement initiatives, and a supportive team culture with opportunities for growth."
Location Filtering:
Matches job locations to user-specified queries.
Hybrid Ranking:
Implements an L2R (Learning-to-Rank) ranker with a BM25 base ranker and cross-encoder features for better relevance.
Interactive Web Interface:
A user-friendly search interface for browsing and navigating job listings.
Prerequisites
Before running the system, ensure the following are installed:

Python (version >= 3.8)
Flask (for serving the web application)
Dependencies listed in requirements.txt (run pip install -r requirements.txt)
HTML/CSS for front-end rendering.
Access to GPT API (optional, for additional data augmentation).
Setup Instructions
Clone this repository:

git clone <repository_url>
cd <repository_directory>
Install dependencies:

pip install -r requirements.txt
Verify the following project structure:

├── app.py             # Main Flask application
├── pipeline.py        # Data processing and ranking pipeline
├── models.py          # Implementation of ranking models
├── web/
    └── home.html      # Front-end template
├── requirements.txt   # Python dependencies
└── README.md          # Documentation
(Optional) Set up API keys for GPT or other external tools in your environment variables:

export OPENAI_API_KEY='your_api_key'
Running the Application
Start the Flask server:

python app.py
Open your browser and navigate to:

http://127.0.0.1:8000/
Use the search bar to input a job query (e.g., "Masters in data science near California").

System Components
1. Back-End
app.py: Handles routing, query processing, and API calls.
pipeline.py: Processes user queries, retrieves job postings, and applies ranking models.
models.py: Contains ranking algorithms, including BM25 and L2R implementations.
2. Front-End
./web/home.html: Provides a user interface for the search engine, including input fields and result displays.
CSS/JavaScript: Enhances interactivity with responsive design and pagination.
Usage
Enter Query: Input a natural language query in the search bar, specifying job criteria such as location, skills, and work culture.
Example:
"Senior software engineer near New York offering flexible remote work."
View Results: Job listings are ranked by relevance and displayed interactively.
Navigate Pages: Use "Next" and "Previous" buttons to browse results.
Click Details: Each job listing includes a link to the original posting for more information.
Query Rules
To ensure accurate results, follow these rules when constructing your query:

Use prepositions for location:
Write near, at, around, or in before the location name.
Example:
"Software engineering roles near California with remote options."
Combine multiple criteria in natural language:
Skills, location, company preferences, and job perks can all be included.
Example:
"Masters in data science with python skills at New York providing mentorship programs."
Acknowledgements
Developed as part of the SI 650 course project at the University of Michigan.
Inspired by traditional IR methods and enhanced with neural ranking techniques.
Front-end designed with PureCSS and custom styles.
