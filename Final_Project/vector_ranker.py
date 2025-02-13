from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np
import torch

class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here
        

        # NOTE: we're going to use the cpu for everything here so if you decide to use a GPU, do not 
        # submit that code to the autograder
        self.biencoder_model = SentenceTransformer(bi_encoder_model_name, device='mps')
        
        # TODO: Also include other necessary initialization code
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid


    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first or an empty list if a query cannot be encoded
            or no results are return
        """
        # NOTE: Do not forget to handle edge cases on the inputs

        # TODO: Encode the query using the bi-encoder


        # TODO: Score the similarity of the query vector and document vectors for relevance
        # Calculate the dot products between the query embedding and all document embeddings
        
        # TODO: Generate the ordered list of (document id, score) tuples

        # TODO: Sort the list so most relevant are first
        if not query or not isinstance(query, str):
            return []
        
        try:
            # Encode the query
            query_embedding = self.biencoder_model.encode(
                query,
                convert_to_tensor=True,  # Get numpy array
                show_progress_bar=False
            )
            
            # Calculate dot product between query embedding and all document embeddings
            # (similarity scores)
            query_embedding = query_embedding.cpu().numpy()
            similarity_scores = np.dot(self.encoded_docs, query_embedding)
            
            # Create list of (doc_id, score) tuples
            doc_scores = [
                (self.row_to_docid[i], float(score)) 
                for i, score in enumerate(similarity_scores)
            ]
            
            # Sort by descending score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores
            
        except Exception as e:
            print(f"Error during query processing: {str(e)}")
            return []

