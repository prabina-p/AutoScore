import reflex as rx
from autoscore import bot_compare, bot_suggests

questionBank = ["Explain a binary search tree", "What is a queue in computer science?", "What is a capacitor"]
questionID = ["Queue", "Binary", "Capacitor"]

class State(rx.State):
    """The app state."""
    #userInput that is passed into grading function
    userInput: str
    #input that is updated in text box
    new_item: str
    #index of which question
    question: str 

    correctness: bool  # Correct -> True
    gpt_feedback: str  # Feedback from GPT
    

    def add_item(self):
        """Add a new item to the todo list."""
        self.userInput = self.new_item
    
    def display_result(self):
        self.correctness = bot_compare(question=self.question, solution="SomeSolution", student_answer=self.userInput)
        self.gpt_feedback = None if self.correctness else bot_suggests(question=self.question, solution="SomeSolution", student_answer=self.userInput)
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
            value=f"Correctness: {State.correctness}, Suggestions: {State.gpt_feedback}",
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