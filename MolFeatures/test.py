from tkinter import Toplevel, Canvas, Scrollbar, Frame, Label, Entry, Tk

class MyApp:
    def __init__(self, master):
        self.master = master
        self.new_window = Toplevel(self.master)
        self.new_window.title("Questions")

        # Create the Canvas widget
        canvas = Canvas(self.new_window)
        canvas.pack(side='left', fill='both', expand=True)

        # Create the vertical Scrollbar and link it to the Canvas
        scrollbar = Scrollbar(self.new_window, orient='vertical', command=canvas.yview)
        scrollbar.pack(side='right', fill='y')

        # Configure the Canvas to use the Scrollbar
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a Frame inside the Canvas to hold the content
        self.content_frame = Frame(canvas)

        # Bind the frame size to the canvas scroll region
        self.content_frame.bind("<Configure>", lambda event: canvas.configure(scrollregion=canvas.bbox("all")))

        # Add the Frame to the Canvas window
        canvas.create_window((0, 0), window=self.content_frame, anchor="nw")

        # Sample list of questions
        questions = ["What is your name?", "How old are you?", "Where do you live?", 
                     "What is your profession?", "What are your hobbies?", 
                     "What is your favorite food?", "What languages do you speak?"]

        # Add questions and entry widgets to the content_frame
        for question in questions:
            # Create a frame for each question to hold the label and entry
            question_frame = Frame(self.content_frame)
            question_frame.pack(pady=5, fill="x")

            # Create a label for the question
            label = Label(question_frame, text=question, wraplength=400)
            label.pack(side="left", padx=5)

            # Create an entry widget for the answer
            entry = Entry(question_frame, width=30)
            entry.pack(side="left", padx=5)

# Example usage
root = Tk()
app = MyApp(root)
root.mainloop()
