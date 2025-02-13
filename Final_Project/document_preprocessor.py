"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""

import nltk, spacy
from nltk.tokenize import RegexpTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from spacy.tokenizer import Tokenizer
from spacy.symbols import ORTH, NORM

nlp = spacy.load("en_core_web_sm")


class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.tokens = []
        self.tokenizer = None
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions
    
    def postprocess(self, input_tokens: list[str]) -> list[str]:
    # MWE 먼저 처리
        if self.multiword_expressions:
            i = 0
            result = []
            while i < len(input_tokens):
                found_mwe = False
                
                # 현재 위치에서 가능한 가장 긴 MWE 찾기
                for mwe in sorted(self.multiword_expressions, key=len, reverse=True):
                    mwe_tokens = mwe.split()
                    mwe_len = len(mwe_tokens)
                    
                    if (i + mwe_len <= len(input_tokens) and 
                        [t.lower() if self.lowercase else t for t in input_tokens[i:i + mwe_len]] == 
                        [t.lower() if self.lowercase else t for t in mwe_tokens]):
                        result.append(mwe)  # 원본 MWE 유지
                        i += mwe_len
                        found_mwe = True
                        break
                
                if not found_mwe:
                    # MWE가 아닌 일반 토큰만 lowercase 적용
                    token = input_tokens[i]
                    result.append(token.lower() if self.lowercase else token)
                    i += 1
                
            return result

        # MWE가 없는 경우 모든 토큰에 lowercase 적용
        if self.lowercase:
            return [token.lower() for token in input_tokens]
        
        return input_tokens
    
    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError('tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.tokenizer = lambda text: text.split()

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        self.tokens = self.tokenizer(text)
        return self.postprocess(self.tokens)

        # pass


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = r'\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.
        https://www.nltk.org/api/nltk.tokenize.RegexpTokenizer.html

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer 
        super().__init__(lowercase, multiword_expressions)
        self.tokenizer = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        https://www.nltk.org/api/nltk.tokenize.RegexpTokenizer.html
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        self.tokens = self.tokenizer.tokenize(text)
        return self.postprocess(self.tokens)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        https://spacy.io/api/tokenizer
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        self.nlp = spacy.load("en_core_web_sm")
            
        self.tokenizer = self.nlp.tokenizer

        special_cases = [
            {ORTH: "Children's"},
            {ORTH: "People's"},
            {ORTH: "New Year's"},
            {ORTH: "Men's"},
            {ORTH: "King's"},
            {ORTH: "Grey's"}
        ]

        # Add each special case
        for case in special_cases:
            self.nlp.tokenizer.add_special_case(case[ORTH], [case])

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        self.tokens = [token.text for token in self.tokenizer(text)]
        return self.postprocess(self.tokens)


# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Apple Silicon(M1/M2)에 최적화된 Doc2Query Augmenter
        """
        if torch.backends.mps.is_available():
            # MPS 메모리 관리 최적화
            torch.mps.set_per_process_memory_fraction(0.9)
            self.device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) device")
            
            # 초기 메모리 정리
            torch.mps.empty_cache()
        else:
            self.device = torch.device("cpu")
            print("MPS not available. Using CPU")

        # 토크나이저와 모델 초기화
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name)
        
        # 모델을 평가 모드로 설정하고 디바이스로 이동
        self.model.eval()
        self.model.to(self.device)
        
        # 모델 워밍업
        self._warmup()

    def _warmup(self):
        """모델 워밍업으로 첫 실행 속도 개선"""
        try:
            with torch.inference_mode():
                dummy_input = self.tokenizer(
                    "warmup text",
                    return_tensors='pt',
                    max_length=50,
                    truncation=True
                ).to(self.device)
                self.model.generate(dummy_input.input_ids, max_length=50)
        except Exception as e:
            print(f"Warmup failed: {e}")

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        MPS 최적화된 쿼리 생성
        """
        document_max_token_length = 400
        top_p = 0.85

        if not document or not isinstance(document, str) or n_queries <= 0:
            return []

        try:
            input_text = prefix_prompt + document
            
            # 추론 모드에서 실행
            with torch.inference_mode():
                input_ids = self.tokenizer.encode(
                    input_text,
                    return_tensors='pt',
                    max_length=document_max_token_length,
                    truncation=True
                ).to(self.device)
                
                outputs = self.model.generate(
                    input_ids,
                    max_length=document_max_token_length,
                    num_return_sequences=n_queries,
                    top_p=top_p,
                    do_sample=True,
                    num_beams=4,
                    early_stopping=True,
                    length_penalty=1.0
                )
                
                # 결과를 CPU로 이동하고 디코딩
                outputs = outputs.cpu()
                query_list = [
                    self.tokenizer.decode(output, skip_special_tokens=True)
                    for output in outputs
                ]

            return query_list
            
        except Exception as e:
            print(f"Error generating queries: {str(e)}")
            return []

    def batch_get_queries(self, documents: list[str], n_queries: int = 5, 
                         batch_size: int = 32, prefix_prompt: str = '') -> list[list[str]]:
        """
        배치 처리로 최적화된 쿼리 생성
        """
        all_queries = []
        
        # 메모리 정리
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        try:
            with torch.inference_mode():
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i + batch_size]
                    batch_texts = [prefix_prompt + doc for doc in batch_docs]
                    
                    # 배치 인코딩
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=400,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # 배치 생성
                    outputs = self.model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_length=400,
                        num_return_sequences=n_queries,
                        top_p=0.85,
                        do_sample=True,
                        num_beams=4,
                        early_stopping=True,
                        length_penalty=1.0
                    )
                    
                    # 결과 처리
                    outputs = outputs.cpu()
                    batch_queries = []
                    
                    for j in range(0, len(outputs), n_queries):
                        doc_queries = outputs[j:j + n_queries]
                        decoded_queries = [
                            self.tokenizer.decode(q, skip_special_tokens=True)
                            for q in doc_queries
                        ]
                        batch_queries.append(decoded_queries)
                    
                    all_queries.extend(batch_queries)
                    
                    # 주기적 메모리 정리
                    if torch.backends.mps.is_available() and i % (batch_size * 10) == 0:
                        torch.mps.empty_cache()
        
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            
        return all_queries