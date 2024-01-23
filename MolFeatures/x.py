# import plotly.graph_objects as go
# import tkinter as tk
# import tkinterweb

# def create_plot():
#     # Create a Plotly figure
#     fig = go.Figure(data=[go.Scatter(x=[1, 2, 3], y=[4, 3, 2])])
    
#     # Convert the figure to HTML
#     plot_html = fig.to_html(include_plotlyjs='cdn', full_html=False)
    
#     return plot_html

# def on_show_plot():
#     plot_html = create_plot()
    
#     # Create a new Tkinter top-level window
#     new_window = tk.Toplevel(root)
#     new_window.title("Plotly Plot")

#     # Embed the plot in the new window
#     html_frame = tkinterweb.HtmlFrame(new_window)
#     html_frame.load_html(plot_html)
#     html_frame.pack(fill="both", expand=True)

# # Tkinter setup
# root = tk.Tk()
# root.title("Tkinter with Plotly Interactive Plot")

# show_plot_button = tk.Button(root, text="Show Plotly Plot", command=on_show_plot)
# show_plot_button.pack(pady=20)

# root.mainloop()
from tkinterweb import HtmlFrame #import the HTML browser
import os
try:
  import tkinter as tk #python3
except ImportError:
  import Tkinter as tk #python2
os.chdir(os.path.dirname(__file__)) #change the working directory to the script directory
root = tk.Tk() #create the tkinter window
frame = HtmlFrame(root) #create the HTML browser
frame.load_website("my_plot.html") #load a website
frame.pack(fill="both", expand=True) #attach the HtmlFrame widget to the parent window
root.mainloop()
