import reflex as rx
from autoscore import query
from autoscore import bot_compare, bot_suggests
from autoscore import create_client_collection, query, predict
from time import perf_counter

questionBank = ["Explain a binary search tree", "What is a queue in computer science?", "What is a capacitor"]
questionID = ["Queue", "Binary", "Capacitor"]
defaultSolutions = {'Queue': "A data structure that can store elements, which has the property that the last item added will be the last to be removed (or first-in-first-out)."}

class State(rx.State):
    """The app state."""
    #userInput that is passed into grading function
    userInput: str
    #input that is updated in text box
    new_item: str
    #index of which question
    question: str 
    # collection_size: int
    source: str  # source of judgement, display as result

    correctness: bool  # Correct -> True
    gpt_feedback: str  # Feedback from GPT

    execution_time: float
    collection_count: float

    def add_item(self):
        """Add a new item to the todo list."""
        self.userInput = self.new_item
    
    def display_result(self):
        # chroma api
        collection = create_client_collection()
        self.collection_count = collection.count()
        self.source = 'ChromaDB'
        start = perf_counter()
        response = query(collection, self.userInput)
        self.correctness = predict(response)[0]
        # print(f"chroma predicted: {self.correctness}")
        # print(f"chroma std-dev: {predict(response)[1]}")
        
        # chroma not amazing on correct preds, gpt api
        if self.correctness == True:
            self.source = "GPT-4"
            # gpt api
            self.correctness = bot_compare(question=self.question, solution=defaultSolutions['Queue'], student_answer=self.userInput)
            end = perf_counter()
            self.gpt_feedback = None if self.correctness else bot_suggests(question=self.question, solution="SomeSolution", student_answer=self.userInput)
        else:
            self.gpt_feedback = bot_suggests(question=self.question, solution="SomeSolution", student_answer=self.userInput)
            end = perf_counter()

        self.execution_time = end - start
        # add response to collection:
        collection.add(
            documents=[self.userInput],
            metadatas=[{"correct": f"{self.correctness}"}],
            ids=[f"id{collection.count()+1}"],
        )
        
        return

def index() -> rx.Component:
    """A view of the todo list.

    Returns:
        The index page of the todo app.
    """
    return rx.container(
        rx.hstack(
            rx.select(questionBank, default_value=questionBank[0], placeholder="Select a question",
                       radius="full", value=State.question, on_change=State.set_question, width="300px"),
            rx.button("Submit answer", on_click=State.display_result()),  
            rx.spacer(),
            # rx.box(State.collection_size, background_color="teal", width="20%"), ##add database stuff here

        ),
        rx.text_area(
            id="new_item",
            placeholder="Your answer here...",
            bg="white",
            value=State.new_item,
            on_change=State.set_new_item,
            on_blur=State.add_item(),
            box_shadow=f"{rx.color('gray', 3, alpha=True)} 0px 1px 4px",
            width="100",
            height="300px",
        ),
        rx.text_area(
            value=f"Correctness: {State.correctness},\n Suggestions: {State.gpt_feedback}, \n Time: {State.execution_time}, Size: {State.collection_count}. ",
            height="300px",
            bg="gray",
            placeholder="Feedback here..."
        ),
        size="2",
        margin_top="5em",
        margin_x="25vw",
        padding="1em",
        border_radius="0.5em",
    )

def gradedResponse() -> rx.Component:
    return rx.container(
        rx.hstack(
            rx.card(State.userInput, width="70"),
            rx.vstack(
                rx.card("Score:"),
                rx.card("100%")
            ),
        ),
        bg="white",
        height="300px",
        margin_x="25vw",
        margin_top="5em",
        padding="1em",
        border_radius="0.5em",
    )


# Create the app and add the state.
app = rx.App()

# Add the index page and set the title.
app.add_page(index, title="Enter your answer", route="/")
app.add_page(gradedResponse, title="Graded response", route="/grade")