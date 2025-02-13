# models.py

'''
Author: Prithvijit Dasgupta

This file contains the base models required by the service to function
'''
from pydantic import BaseModel

class QueryModel(BaseModel):
    query:str

# class SearchResponse(BaseModel):
#     id: int
#     docid: int
#     score: float

class SearchResponse(BaseModel):
    id: int
    docid: int      
    score: float
    title: str
    company: str
    location: str
    job_url: str

class PaginationModel(BaseModel):
    prev: str
    next: str

class APIResponse(BaseModel):
    results: list[SearchResponse]
    page: PaginationModel | None

class ExperimentResponse(BaseModel):
    ndcg: float
    query: str

class BaseSearchEngine():
    def __init__(self, path: str) -> None:
        pass

    def index(self):
        pass

    def search(self, query: str) -> list[SearchResponse]:
        pass