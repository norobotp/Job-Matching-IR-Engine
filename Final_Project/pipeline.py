# pipeline.py is the main file that initializes the search engine and performs a search query.
'''
Author: Prithvijit Dasgupta
Modified by: Zim Gong
This file is a template code file for the Search Engine.
'''
from collections import defaultdict
import pandas as pd
from document_preprocessor import RegexTokenizer
from indexing import Indexer, IndexType, BasicInvertedIndex
from ranker import *
from models import BaseSearchEngine, SearchResponse
from l2r import L2RRanker, L2RFeatureExtractor
from network_features import NetworkFeatures
from typing import List
from ranker import Ranker, BM25, CrossEncoderScorer
from vector_ranker import VectorRanker
from location_filter import LocationFilteredRanker
import csv
from typing import Dict


# Define constants for file paths
DATA_PATH = './'
STOPWORD_PATH = DATA_PATH + 'stopwords.txt'
UPDATED_JOB_DOCID_PATH = DATA_PATH + 'updated_jobs_docid.csv'
LOCATION_DATA_PATH = DATA_PATH + 'updated_jobs_location_final.csv' 
TRAIN_DATA_PATH = DATA_PATH + 'big_train_data_new_adjusted.csv'
MAIN_INDEX_PATH = './augmented_job_description_index'
TITLE_INDEX_PATH = './combined_job_title_indexing_14956'
ENCODED_DOCS_PATH = './encoded_jobs.npy'


class JobSearchEngine(BaseSearchEngine):
    def __init__(self, max_docs: int = -1, ranker_type: str = 'combined') -> None:
        """
        Initialize job search engine with specified parameters
        
        Args:
            max_docs (int): Maximum number of documents to process (-1 for all)
            ranker_type (str): Type of ranker to use ('combined', 'vector', or 'bm25')
        """
        print('Initializing Job Search Engine...')
        
        # Load all required resources
        self.load_resources()
        
        # Initialize rankers based on type
        self.ranker_type = ranker_type
        self.initialize_rankers()
        
        print('Job Search Engine initialized!')

    @staticmethod
    def load_location_data(filepath: str) -> Dict[str, str]:
        """
        Load location data from CSV file
        
        Args:
            filepath (str): Path to location data CSV
                
        Returns:
            Dict: Mapping of doc_ids to location information
        """
        df = pd.read_csv(filepath)
        return dict(zip(df['docid'].astype(int), df['formatted_location']))
    
    def load_resources(self):
        """Load all required resources and data"""
        # Load stopwords
        print('Loading stopwords...')
        with open(STOPWORD_PATH, 'r', encoding='utf-8') as f:
            self.stopwords = set(line.strip() for line in f)

        # Initialize preprocessor
        print('Initializing preprocessor...')
        self.preprocessor = RegexTokenizer(r'\w+')

        # Load indices
        print('Loading indices...')
        self.main_index = BasicInvertedIndex()
        self.main_index.load(MAIN_INDEX_PATH)
        
        self.title_index = BasicInvertedIndex()
        self.title_index.load(TITLE_INDEX_PATH)

        # Load job data
        print('Loading job data...')
        self.job_data = pd.read_csv(UPDATED_JOB_DOCID_PATH)
        self.doc_location_dict = self.load_location_data(LOCATION_DATA_PATH)
        
        # Create raw text dictionary for cross encoder
        self.raw_text_dict = dict(zip(
            self.job_data['docid'], 
            self.job_data['description']
        ))


    def initialize_rankers(self):
        """Initialize all ranking models"""
        print('Initializing rankers...')
        
        # Initialize BM25
        print('Setting up BM25...')
        self.bm25_scorer = BM25(self.main_index)
        self.bm25_ranker = Ranker(
            self.main_index,
            self.preprocessor,
            self.stopwords,
            self.bm25_scorer
        )

        # Initialize Vector Ranker
        print('Setting up Vector Ranker...')
        encoded_docs = np.load(ENCODED_DOCS_PATH)
        row_to_docid = list(self.job_data['docid'])
        self.vector_ranker = VectorRanker(
            'sentence-transformers/msmarco-MiniLM-L12-cos-v5',
            encoded_docs,
            row_to_docid
        )
        

        # Initialize Cross Encoder
        print('Setting up Cross Encoder...')
        self.ce_scorer = CrossEncoderScorer(raw_text_dict=self.raw_text_dict)
        # self.ce_scorer_diff = CrossEncoderScorer(raw_text_dict=self.raw_text_dict,  cross_encoder_model_name='cross-encoder/nli-deberta-v3-base')

        

        # Initialize L2R Feature Extractor
        print('Setting up L2R Feature Extractor...')
        self.l2r_feature_extractor = L2RFeatureExtractor(
            self.main_index,
            self.title_index,
            {},  # Empty doc category info
            self.preprocessor,
            self.stopwords,
            set(),  # Empty recognized categories
            {},  # Empty network features
            self.ce_scorer,
            self.doc_location_dict
        )

        # Initialize L2R Rankers
        print('Setting up L2R Rankers...')
        self.l2r_bm25 = L2RRanker(
            self.main_index,
            self.title_index,
            self.preprocessor,
            self.stopwords,
            self.bm25_ranker,
            self.l2r_feature_extractor
        )

        self.l2r_vector = L2RRanker(
            self.main_index,
            self.title_index,
            self.preprocessor,
            self.stopwords,
            self.vector_ranker,
            self.l2r_feature_extractor
        )

        # Initialize Location Filtered Rankers
        print('Setting up Location Filtered Rankers...')
        self.location_bm25 = LocationFilteredRanker(
            self.l2r_bm25,
            self.doc_location_dict
        )
        self.location_vector = LocationFilteredRanker(
            self.l2r_vector,
            self.doc_location_dict
        )

        # Train L2R models
        print('Training L2R models...')
        self.l2r_bm25.train(TRAIN_DATA_PATH)
        self.l2r_vector.train(TRAIN_DATA_PATH)

    def search(self, query: str) -> List[SearchResponse]:
        """
        Perform search with the given query and return results
        """
        # Select ranker based on type
        if self.ranker_type == 'combined':
            results = self.location_bm25.query(query)
        elif self.ranker_type == 'vector':
            results = self.location_vector.query(query)
        else:
            results = self.bm25_ranker.query(query)

        # Format results
        search_responses = []
        for idx, (doc_id, score) in enumerate(results[:100]):  # Limit to top 100
            job_info = self.get_job_info(int(doc_id))  # doc_id를 int로 변환
            search_responses.append(
                SearchResponse(
                    id=idx+1,
                    docid=int(doc_id),  # docid를 int로 명시적 변환
                    score=float(score),
                    title=job_info['title'],
                    company=job_info['company'],
                    location=job_info['location'],
                    job_url=job_info['job_url']
                )
            )

        return search_responses

    def get_job_info(self, doc_id: int) -> dict:
        """
        Get job information for the given doc_id
        """
        row = self.job_data[self.job_data['docid'] == doc_id]
        if not row.empty:
            return {
                'title': str(row['title'].iloc[0]) if not pd.isna(row['title'].iloc[0]) else 'No Title',
                'company': str(row['company'].iloc[0]) if not pd.isna(row['company'].iloc[0]) else 'Unknown Company',
                'location': str(row['location'].iloc[0]) if not pd.isna(row['location'].iloc[0]) else 'Location Unknown',
                'job_url': str(row['job_url'].iloc[0]) if not pd.isna(row['job_url'].iloc[0]) else '#'
            }
        return {
            'title': 'No Title',
            'company': 'Unknown Company',
            'location': 'Location Unknown',
            'job_url': '#'
        }

def initialize(max_docs: int = -1, ranker_type: str = 'combined') -> JobSearchEngine:
    """
    Initialize and return the JobSearchEngine object
    
    Args:
        max_docs (int): Maximum number of documents to process
        ranker_type (str): Type of ranker to use
        
    Returns:
        JobSearchEngine: Initialized search engine
    """
    return JobSearchEngine(max_docs=max_docs, ranker_type=ranker_type)


if __name__ == '__main__':
    # Test the search engine
    engine = initialize()
    
    # Example search query
    query = "senior software engineer with python experience in New York"
    results = engine.search(query)
    
    # Print top 5 results
    print("\nTop 5 Results:")
    for result in results[:5]:
        print(f"\nRank: {result.id}")
        print(f"Title: {result.title}")
        print(f"Company: {result.company}")
        print(f"Location: {result.location}")
        print(f"Score: {result.score:.4f}")
        print(f"URL: {result.job_url}")