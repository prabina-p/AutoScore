import chromadb
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


### chroma db implementation:

def create_client_collection():
    ''' 
    Create a client and collection using ChromaDB
    @return: the client created
    '''
    # return chromadb.Client('http://localhost:5000')
    try:
        client = chromadb.PersistentClient(path="./database")
        # collection = client.create_collection("BST_question", embedding_function=huggingface_ef)
        collection = client.create_collection(
            name="Queue_question",
            metadata={"hnsw:space": "cosine"} # l2 is the default
        )
    except:
        client = chromadb.PersistentClient(path="./database")
        collection = client.get_collection("Queue_question")
    return collection

def read_data(path):
    '''
    Read the data from the path
    @param path: the path to the data
    @return: the data'''
    data = pd.read_csv(path).iloc[:, 1:]
    return data

def add_to_collection(collection, df_corr, df_incorr):
    '''
    Add the student answers to the collection
    @param collection: the collection to add the student answers to
    @param df_corr: the dataframe of correct student answers
    @param df_incorr: the dataframe of incorrect student answers
    '''
    # add correct responses
    l = df_corr['student_answer'].tolist()
    ids = [f"id{i}"for i in range(len(l))]
    collection.add(
        documents=df_corr['student_answer'].tolist(),
        metadatas=[{"correct": "True"} for _ in range(len(l))],
        ids=ids,
    )
    
    # add incorrect responses
    l2 = df_incorr['student_answer'].tolist()

    ids = [f"id{i}"for i in range(len(l), len(l2)+len(l))]
    collection.add(
        documents=df_incorr['student_answer'].tolist(),
        metadatas=[{"correct": "False"} for _ in range(len(l2))],
        ids=ids,
    )

def query(collection, student_answer, k):
    '''
    Query the collection for the student answer
    @param collection: the collection to query
    @param student_answer: the student answer to query for
    @param k: the number of responses to return
    @return: the response from the query
    '''
    response = collection.query(
        document=student_answer,
        k=k,
    )
    return response

def predict(response_json):
    '''
    Return the prediction based on k-nn average vote
    @param response_json: the response from the query
    @return: the prediction
    '''
    correct_sum = [x['correct'] for x in response_json['metadatas'][0]].count('True')
    weighted_vote = (correct_sum/3) >= 0.5
    return weighted_vote

    
    
    