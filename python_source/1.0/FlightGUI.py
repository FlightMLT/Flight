from tkinter import *
from PIL import Image, ImageTk, ImageSequence
import itertools
from pathlib import Path

root = Tk()

# Get the directory of the script
script_dir = Path(__file__).parent

# Dynamically create paths for the icon and gif
icon_path = script_dir / "icon.png"
gif_path = script_dir / "loading.gif"

# Convert Path objects to strings
icon_path_str = str(icon_path)
gif_path_str = str(gif_path)

# Load the icon
icon = PhotoImage(file=icon_path_str)
root.iconphoto(False, icon)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.title("Flight")
root.configure(background="black")

# Load and process the GIF
gif_image = Image.open(gif_path_str)
frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(gif_image)]
frame_count = len(frames)

gif_label = Label(root, background="black")
gif_label.place(relx=0.5, rely=0.5, anchor="center")  # Center the label

def update_gif(ind):
    frame = frames[ind]
    gif_label.config(image=frame)
    root.after(100, update_gif, (ind + 1) % frame_count)

update_gif(0)

root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))
root.bind("<F11>", lambda event: root.attributes("-fullscreen", True))
root.geometry(f"{screen_width}x{screen_height}+0+0")
root.attributes('-fullscreen', False)

root.mainloop()
