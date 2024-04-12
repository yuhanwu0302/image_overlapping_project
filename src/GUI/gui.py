import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps

class ImageRotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Rotator")
        
        self.canvas = tk.Canvas(self.master, width=512, height=512, bg="white")
        self.canvas.pack()

        self.open_button = tk.Button(self.master, text="Open", command=self.open_file)
        self.open_button.pack()

        self.save_button = tk.Button(self.master, text="Save", command=self.save_rotated_image)
        self.save_button.pack()

        
        self.canvas.bind("<ButtonPress-1>", self.start_rotation)
        self.canvas.bind("<B1-Motion>", self.rotate_image)

        self.image = None
        self.tk_image = None
        self.rotation_made = False
        self.angle = 0

    def open_file(self):
        filename = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if filename:
            self.load_image(filename)

    def load_image(self, filename):
       
        self.image = Image.open(filename)
        self.image.thumbnail((512, 512)) 
        self.tk_image = ImageTk.PhotoImage(self.image)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def start_rotation(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def rotate_image(self, event):
        if self.image:
            angle = (event.x - self.start_x) * 0.5
            self.angle = (self.angle + angle) % 360  # Update rotation angle
            rotated_image = self.image.rotate(self.angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
            self.tk_image = ImageTk.PhotoImage(rotated_image)
            self.canvas.delete("all")  # Delete previous image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
            
            # Set the flag to indicate rotation has been made
            self.rotation_made = True


    def save_rotated_image(self):
        if self.rotation_made:
            filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if filename:
                # Create a blank 512x512 image with white background
                blank_image = Image.new("RGB", (512, 512), color="white")
                # Rotate the original image
                rotated_image = self.image.rotate(self.angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
                # Paste the rotated image onto the blank image
                blank_image.paste(rotated_image, (0, 0))
                # Save the rotated and pasted image as PNG
                blank_image.save(filename)
                self.rotation_made = False  # Reset rotation flag after saving
        else:
            messagebox.showwarning("Warning", "No rotation has been made yet.")

def main():
    root = tk.Tk()
    app = ImageRotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
