import lightgbm

from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
from ranker import *
import csv
import numpy as np
from tqdm.auto import tqdm
from typing import List
from location_filter import LocationFilter


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** hw3 modified **
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any arguments that are needed as fields of this class
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.ranker = ranker
        self.feature_extractor = feature_extractor
        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.model = LambdaMART()


    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # TODO: for each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features

        if not query_to_document_relevance_scores:
            return X, y, qgroups  # 빈 리스트들 반환
    
        for query, doc_relevance_scores in query_to_document_relevance_scores.items():
            query_tokens = self.document_preprocessor.tokenize(query)
            
            query_tokens_stopwords_removed = []
            if self.stopwords:
                for token in query_tokens:
                    if token not in self.stopwords:
                        query_tokens_stopwords_removed.append(token)
            
            # query_tokens = query_tokens_stopwords_removed

            doc_term_counts_dict = self.accumulate_doc_term_counts(self.document_index, query_tokens_stopwords_removed)
            title_term_counts_dict = self.accumulate_doc_term_counts(self.title_index, query_tokens_stopwords_removed)
            

            query_doc_counts = 0
            for docid, relevance in doc_relevance_scores:
                doc_word_counts = doc_term_counts_dict.get(docid, {})
                title_word_counts = title_term_counts_dict.get(docid, {})
                #initialize the term counts for the document and title with the query_tokens
                
                features = self.feature_extractor.generate_features(docid, doc_word_counts, title_word_counts, query_tokens)
                                                            
                X.append(features)
                y.append(relevance)
                query_doc_counts += 1
        
            if query_doc_counts > 0:  # 문서가 하나라도 처리된 경우만 추가
                qgroups.append(query_doc_counts)

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        accum_doc_term_counts = {}

        query_parts = list(set(query_parts))

        for token_term in query_parts:
            for docid, frequency in index.get_postings(token_term):
                if docid not in accum_doc_term_counts:
                    #initialize the term counts for the document with the query_tokens
                    accum_doc_term_counts[docid] = {}
                accum_doc_term_counts[docid][token_term] = frequency

        return accum_doc_term_counts
    

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """

        query_to_doc_rel_scores = {}
        with open(training_data_filename, 'r', newline='', errors='ignore') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                query = row['query']
                doc_id = int(row['doc_id'])
                relevance = int(float(row['Relevance_Score']) * 2)

                if query not in query_to_doc_rel_scores:
                    #initialize the list of documents and relevance scores for the query
                    query_to_doc_rel_scores[query] = []
                query_to_doc_rel_scores[query].append((doc_id, relevance))

        X, y, qgroups = self.prepare_training_data(query_to_doc_rel_scores)

        self.model.fit(X, y, qgroups)

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        

        return self.model.predict(X)
    

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        

        if not query or not isinstance(query, str):
            return []
        
        
        tokenized_query = self.document_preprocessor.tokenize(query)
        
        if not tokenized_query:
            return []

        query_word_counts = {}
        stopword_count = 0

        for word in tokenized_query:
            if word in self.stopwords:
                stopword_count += 1
            else:
                query_word_counts[word] = query_word_counts.get(word, 0) + 1

        if stopword_count > 0:
            query_word_counts[None] = stopword_count

        doc_word_counts = self.accumulate_doc_term_counts(self.document_index, query_word_counts.keys())

        title_word_counts = self.accumulate_doc_term_counts(self.title_index, query_word_counts.keys())

        
        if not doc_word_counts:
            return []

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        # pass these doc-term-counts to functions later, so we need the accumulated representations



        # TODO: Accumulate the documents word frequencies for the title and the main body

        

        # TODO: Score and sort the documents by the provided scrorer for just the document's main text (not the title)
        
        # This ordering determines which documents we will try to *re-rank* using our L2R model


        sorted_score = self.ranker.query(query)



        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking

        top_150 = sorted_score[:150]

        if not top_150:
            return []

        # TODO: Construct the feature vectors for each query-document pair in the top 100

        X = []
        X_docid = []
        with tqdm(total=len(top_150), desc=f"Ranking top {len(top_150)}") as pbar:
            for docid, _ in top_150:
                try:
                    features = self.feature_extractor.generate_features(
                        docid, 
                        doc_word_counts.get(docid, {}), 
                        title_word_counts.get(docid, {}), 
                        tokenized_query
                    )
                    
                    X.append(features)
                    X_docid.append(docid)
                    pbar.update(1)
                except Exception as e:
                    print(f"\nFailed at docid {docid}: {str(e)}")
                    pbar.update(1)
                    continue
        
        if not X:
            return []

        # TODO: Use your L2R model to rank these top 100 documents

        if not hasattr(self.model.model, 'fitted_') or not self.model.model.fitted_:
            # 학습되지 않은 경우 초기 랭킹 결과 반환
            print("Model has not been trained yet. Returning initial ranking.")
            return sorted_score
        

        predicted_scores = self.predict(X)
        
        predicted_scores_zip = zip(X_docid, predicted_scores)


        # TODO: Sort posting_lists based on scores
        
        
        sorted_predicted_scores = sorted(predicted_scores_zip, key=lambda x: x[1], reverse=True)


        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked

        ranked_docs = sorted_predicted_scores + [(docid, score) for docid, score in sorted_score[150:]]

        # print("Ranked docs",ranked_docs)


        # TODO: Return the ranked documents
        return ranked_docs


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer, doc_locations: dict[int, str]) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.document_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.recognized_categories = recognized_categories
        self.docid_to_network_features = docid_to_network_features
        self.ce_scorer = ce_scorer

        self.doc_locations = doc_locations
        self.location_filter = LocationFilter()


        # TODO: For the recognized categories (i.e,. those that are going to be features), considering
        # how you want to store them here for faster featurizing
        self.doc_category_features = {}
        for docid in doc_category_info:
            category_index = []
            for i, category in enumerate(recognized_categories):
                if category in doc_category_info[docid]:
                    category_index.append(1)
                else:
                    category_index.append(0)
            self.doc_category_features[docid] = category_index
        
        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring.
        self.bm25_scorer = BM25(self.document_index)
        self.pivoted_normalization_scorer = PivotedNormalization(self.document_index)

    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """

        doc_metadata = self.document_index.get_doc_metadata(docid)
    
        if 'length' not in doc_metadata:
            raise KeyError(f"Document with docid {docid} has no 'length' key in metadata.")
        
        return doc_metadata['length']

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)['length']

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """
        tf = 0
        for word in query_parts:
            if word not in self.stopwords:
                doc_term_count = word_counts.get(word, 0)
                tf += np.log(1 + doc_term_count)
        return tf

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_word_counts = {}
        stopword_count = 0

        for word in query_parts:
            if word in self.stopwords:
                stopword_count += 1
            else:
                query_word_counts[word] = query_word_counts.get(word, 0) + 1

        if stopword_count > 0:
            query_word_counts[None] = stopword_count


        tf_idf_scorer = TF_IDF(index)
        tf_idf_score = tf_idf_scorer.score(docid, word_counts, query_word_counts)

        return tf_idf_score
        

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        
        query_word_counts = {}
        stopword_count = 0

        for word in query_parts:
            if word in self.stopwords:
                stopword_count += 1
            else:
                query_word_counts[word] = query_word_counts.get(word, 0) + 1

        if stopword_count > 0:
            query_word_counts[None] = stopword_count


        # print("doc_word_counts for BM25 : ", doc_word_counts)
        bm25_score = self.bm25_scorer.score(docid, doc_word_counts, query_word_counts)

        return bm25_score

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        query_word_counts = {}
        stopword_count = 0

        for word in query_parts:
            if word in self.stopwords:
                stopword_count += 1
            else:
                query_word_counts[word] = query_word_counts.get(word, 0) + 1

        if stopword_count > 0:
            query_word_counts[None] = stopword_count

        
        pivoted_normalization_score = self.pivoted_normalization_scorer.score(docid, doc_word_counts, query_word_counts)
        
        return pivoted_normalization_score
    
    def get_own_feature(self, query_parts: list[str]) -> float:
        query_word_counts = {}
        for word in query_parts:
            if word in query_word_counts:
                query_word_counts[word] += 1
            else:
                query_word_counts[word] = 1
        
        feature_score = len(query_word_counts.keys()) / len(query_parts)
        return feature_score


    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """        
        if not hasattr(self, 'ce_scorer') or self.ce_scorer is None:
            return 0.0
        return self.ce_scorer.score(docid, query)
    def get_query_term_ratio(self, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """쿼리 텀이 문서에 얼마나 포함되어 있는지의 비율"""
        query_terms = set([term for term in query_parts if term not in self.stopwords])
        if not query_terms:
            return 0.0
        matched_terms = sum(1 for term in query_terms if term in doc_word_counts)
        return matched_terms / len(query_terms)

    def get_query_term_statistics(self, doc_word_counts: dict[str, int], query_parts: list[str]) -> List[float]:
        """쿼리 텀들의 통계적 특성"""
        term_counts = [doc_word_counts.get(term, 0) for term in query_parts if term not in self.stopwords]
        if not term_counts:
            return [0.0, 0.0, 0.0, 0.0]
        
        return [
            np.mean(term_counts),
            np.std(term_counts) if len(term_counts) > 1 else 0,
            np.max(term_counts),
            np.min(term_counts)
        ]

    def get_term_density(self, doc_word_counts: dict[str, int], query_parts: list[str], doc_length: int) -> float:
        """쿼리 텀들의 밀도"""
        if doc_length == 0:
            return 0.0
        total_query_terms = sum(doc_word_counts.get(term, 0) for term in query_parts if term not in self.stopwords)
        return total_query_terms / doc_length

    def calculate_query_title_similarity(self, title_word_counts: dict[str, int], query_parts: list[str], 
                                    title_length: int) -> float:
        """제목과 쿼리의 유사도"""
        if title_length == 0:
            return 0.0
        query_terms = set([term for term in query_parts if term not in self.stopwords])
        if not query_terms:
            return 0.0
        
        matching_terms = sum(1 for term in query_terms if term in title_word_counts)
        return matching_terms / len(query_terms)
    
    def get_location_match(self, docid: int, query_parts: list[str]) -> int:
        """Checks if query location matches the document location."""
        state_from_query = self.location_filter.extract_state_from_query(' '.join(query_parts))
        doc_location = self.doc_locations.get(docid, '')

        if not state_from_query or not doc_location:
            return 0
        
        # Extract state from document location
        doc_state = doc_location.split(',')[0].strip().upper()
        return 1 if state_from_query == doc_state else 0

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                     title_word_counts: dict[str, int], query_parts: list[str]) -> list:
        """
        Enhanced feature generation
        """
        feature_vector = []
        
        # Basic features
        doc_length = self.get_article_length(docid)
        title_length = self.get_title_length(docid)
        query_length = len([term for term in query_parts if term not in self.stopwords])
        
        # Original scoring features
        tf_doc = self.get_tf(self.document_index, docid, doc_word_counts, query_parts)
        tf_idf_doc = self.get_tf_idf(self.document_index, docid, doc_word_counts, query_parts)
        tf_title = self.get_tf(self.title_index, docid, title_word_counts, query_parts)
        tf_idf_title = self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts)
        bm25 = self.get_BM25_score(docid, doc_word_counts, query_parts)
        pivoted_normalization = self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts)
        
        # Advanced features
        query_coverage = self.get_query_term_ratio(doc_word_counts, query_parts)
        term_stats = self.get_query_term_statistics(doc_word_counts, query_parts)
        term_density = self.get_term_density(doc_word_counts, query_parts, doc_length)
        title_similarity = self.calculate_query_title_similarity(title_word_counts, query_parts, title_length)
        
        # Cross encoder score
        cross_encoder_score = self.get_cross_encoder_score(docid, ' '.join(query_parts))
        
        location_match = self.get_location_match(docid, query_parts)

        # Combine all features
        feature_vector.extend([
            doc_length,
            title_length,
            query_length,
            tf_doc,
            tf_idf_doc,
            tf_title,
            tf_idf_title,
            bm25,
            pivoted_normalization,
            cross_encoder_score,
            query_coverage,
            term_density,
            title_similarity,
            # location_match
        ])
        
        # Add term statistics
        feature_vector.extend(term_stats)
        
        # Add ratio features
        if doc_length > 0:
            feature_vector.append(title_length / doc_length)  # title-document length ratio
        else:
            feature_vector.append(0.0)
        
        if query_length > 0:
            feature_vector.append(title_similarity / query_length)  # normalized title similarity
        else:
            feature_vector.append(0.0)

        return feature_vector



class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 10,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.04,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": -1,
        }

        if params:
            default_params.update(params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.model = lightgbm.LGBMRanker(**default_params)

    def fit(self,  X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        self.model.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # TODO: Generating the predicted values using the LGBMRanker
        return self.model.predict(featurized_docs)

