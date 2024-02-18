import chromadb
import pandas as pd
import numpy as np
import sklearn
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt



### thresholding implementation:




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


### chat gpt implementation:
API_KEY = "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"
client = OpenAI(api_key=API_KEY)


# True or False
def bot_compare(question: str, solution: str, student_answer: str, special_requirements: str = None, strictness: int = 5) -> bool:
	"""
	Compare the student's answer with the standard answer under the context provided by the question.
	:param question: The question.
	:param solution: The standard answer.
	:param student_answer: The student's answer.
	:param special_requirements: Special requirements about the grading from the instructor.
	:param strictness: A strictness scale out of 10.
	:return: True if the student's answer is equivalent to the standard answer, False otherwise.
	"""
	prompt_1 = f"""Your role is a grader for free response question of a class, 
					and the only thing you do is to do is to compare the student 
					answer with a given standard answer under the context provided by the question. Since
					 you are the grader, there could be some special requirements about the grading
					from the instructor that you must follow when doing your grading.  You only output 1 or 0, 
					according to whether they are equivalent or not under a strictness scale out of 10. You will 
					be given the student's answer as the user's input. \n 
					- The question is {question}. \n 
					- The standard answer is {solution}. \n 
					- Strictness: {strictness} \n
					- Special Requirement: {special_requirements}"""

	response = client.chat.completions.create(
		model="gpt-4",
		messages=[
		{
			"role": "system",
			"content": prompt_1
		},
		{
			"role": "user",
			"content": student_answer
		}
		],
		temperature=0.3,
		max_tokens=64,
		top_p=1
	)
	actual_response = response.choices[0].message.content
	# print(f"Response: {bool(actual_response)}")
	return bool(int(actual_response))

def bot_suggests(question: str, solution: str, student_answer: str, special_requirements: str = None) -> str:
	"""If the student's answer is not equivalent to the standard answer, the bot will suggest the student to improve his/her answer.
	"""

	prompt = f"""A student's free response answer is wrong for a question. You want to provide suggestions for student to improve 
				his/her answer. The suggestions should guide his thinking process. 
				You will be given the question prompt, solution, and student's answer as the user's input. Instructors 
				might have some special instructions about the question. \n
				- The question is {question}. \n
				- The standard answer is {solution}. \n
				- The student's answer is {student_answer}. \n
				- Special Requirement: {special_requirements}"""
	# Create chat gpt session and send the request
	response = client.chat.completions.create(
		model="gpt-4",
		messages=[
		{
			"role": "system",
			"content": prompt
		}
		],
		temperature=0.3,
		max_tokens=128,
		top_p=1
	)
	actual_response = response.choices[0].message.content
	return actual_response
	



    
    
    