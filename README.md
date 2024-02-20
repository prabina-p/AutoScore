# AutoScore

AutoScore aims to simplify the free response grading pipeline faced by instructors by utilizing ML methodologies to predict whether or not a student's answer is deemed correct.


## Inspiration

With a great amount of experience teaching and tutoring at the university level, we knew there was a lot to be desired in the grading experience for both students and instructors. We wished that there was a way students could receive feedback quickly and overworked instructors could focus their attention on more impactful things than grading. As a result, we decided to build a tool that would auto grade short answer response while allowing a high degree of accuracy and customization.


## What it does

Given a student response, our program analyzes the similarity to teacher provided answers. Furthermore, it uses GPT to provide quick feedback for students.


## How we built it

We used ChromaDB to handle our vector database operations and GPT4 to provide feedback for students. For our front-end, we used Reflex as our full-stack solution.


## Demo

* <a href="https://youtu.be/S7EiVUkjzv4" target="_blank">Click for demo video!</a>
* <a href="https://docs.google.com/presentation/d/1RBxTbz7ldAjXelZbIjB6Gruq3qAe4EMrT1bdpLyCLMM/edit?usp=sharing" target="_blank">Click for slides!</a>


## Built With

* chroma
* gpt-4
* python
* reflex
* scikit-learn

