import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from rv_background.remove_background import process_images

class ImageRotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Rotator")

        self.create_menu()
        
        self.canvas = tk.Canvas(self.master, width=512, height=512, bg="white")
        self.canvas.pack()

        self.open_button = tk.Button(self.master, text="Open", command=self.open_file)
        self.open_button.pack()

        self.save_button = tk.Button(self.master, text="Save", command=self.save_rotated_image)
        self.save_button.pack()
        
        self.rotation_entry = tk.Entry(self.master)
        self.rotation_entry.pack()
        
        self.rotate_button = tk.Button(self.master, text="Rotate", command=self.rotate_by_entry)
        self.rotate_button.pack()

        self.canvas.bind("<ButtonPress-1>", self.start_rotation)
        self.canvas.bind("<B1-Motion>", self.rotate_image)

        self.position_button = tk.Button(self.master, text="Track Position", command=self.toggle_position_tracking)
        self.position_button.pack()
        
        self.position_tracking = False  # 初始時不啟用定位追蹤
        self.clicked_marker = None  # 初始化標記變數
        
        self.position_label = tk.Label(self.master, text="Position on canvas: x:  y: ")
        self.position_label.pack()

        self.image = None
        self.tk_image = None
        self.angle = 0

    def create_menu(self):
        menubar = tk.Menu(self.master)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_separator()
        
        remove_background_menu = tk.Menu(file_menu, tearoff=0) 
        remove_background_menu.add_command(label="Open Image", command=self.open_image)
        remove_background_menu.add_command(label="Open Directory", command=self.open_dir)
        file_menu.add_cascade(label="Remove Background", menu=remove_background_menu)

        menubar.add_cascade(label="File", menu=file_menu)

        self.master.config(menu=menubar)

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
        if not self.position_tracking:
            self.start_rotation_action(event)

    def start_rotation_action(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def rotate_image(self, event):
        if not self.position_tracking and self.image:
            rotation_input = self.rotation_entry.get()
            try:
                angle_from_entry = float(rotation_input)
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid rotation angle.")
                return
        
            # Calculate rotation angle from mouse movement
            self.calculate_rotation_angle(angle_from_entry, event)

    def calculate_rotation_angle(self, angle_from_entry, event):
        angle_from_mouse = (event.x - self.start_x) * 0.5
        
        # Combine rotation angles from entry and mouse movement
        self.angle = angle_from_entry + angle_from_mouse
        
        rotated_image = self.image.rotate(self.angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
        self.tk_image = ImageTk.PhotoImage(rotated_image)
        self.canvas.delete("all")  # Delete previous image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def rotate_by_entry(self):
        if not self.position_tracking:
            rotation_input = self.rotation_entry.get()
            try:
                angle = float(rotation_input)
                # Update current angle
                self.angle += angle
            except ValueError:
                messagebox.showerror("Error", "Please enter a valid rotation angle.")
                return
            
            # Rotate the image using the current angle
            rotated_image = self.image.rotate(self.angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
            self.tk_image = ImageTk.PhotoImage(rotated_image)
            self.canvas.delete("all")  # Delete previous image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_rotated_image(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filename:
            blank_image = Image.new("RGB", (512, 512), color="white")
            rotated_image = self.image.rotate(self.angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
            blank_image.paste(rotated_image, (0, 0))
            blank_image.save(filename)

    def open_image(self):
        # 選取單一或多個圖片
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png")])
        if file_paths:
            for file_path in file_paths:
                if not process_images(file_path):  
                    messagebox.showerror("Error", f"Failed to remove background for {file_path}")
                else:
                    messagebox.showinfo("Background Removed", f"Background removed successfully for {file_path}")

    def open_dir(self):
        # 選取一個資料夾
        folder_path = filedialog.askdirectory()
        if folder_path:
            if not process_images(folder_path):  
                messagebox.showerror("Error", f"Failed to remove background for images in {folder_path}")
            else:
                messagebox.showinfo("Background Removed", f"Background removed successfully for images in {folder_path}")

    def toggle_position_tracking(self):
        self.position_tracking = not self.position_tracking
        if self.position_tracking:
            self.position_button.config(text="Stop Position Tracking")
            self.canvas.bind("<Button-1>", self.canvas_click)  # 開啟定位追蹤功能
        else:
            self.position_button.config(text="Track Position")
            self.canvas.unbind("<Button-1>")  # 停止定位追蹤功能

    def canvas_click(self, event):
        # 在 512x512 畫布中的坐標
        canvas_x = event.x
        canvas_y = event.y
        print(f"Position on canvas: ({canvas_x}, {canvas_y})")
        
        # 如果已經有標記存在，則刪除它
        if self.clicked_marker:
            self.canvas.delete(self.clicked_marker)
        
        self.position_label.config(text=f"Position on canvas: x: {canvas_x} y: {canvas_y}")
        
        # 在點擊的位置繪製一個紅色圓點作為標記
        self.clicked_marker = self.canvas.create_oval(canvas_x - 3, canvas_y - 3, canvas_x + 3, canvas_y + 3, fill="red")


def main():
    root = tk.Tk()
    app = ImageRotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
