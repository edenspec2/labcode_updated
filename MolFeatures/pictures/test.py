import tkinter as tk
from PIL import Image, ImageTk
import os
os.chdir("pictures")
# Create the main window
root = tk.Tk()
root.title("Image Display")

# Load the image using Pillow
image = Image.open("rings.png")  # Replace 'your_image.png' with your image file's path
photo = ImageTk.PhotoImage(image)

# Display the image
label = tk.Label(root, image=photo)
label.image = photo  # Keep a reference!
label.pack()

# Run the application
root.mainloop()
