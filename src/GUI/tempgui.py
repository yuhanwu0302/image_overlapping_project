import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import csv

class ImageRotator:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Rotator")

        self.create_menu()
        
        self.canvas = tk.Canvas(self.master, width=512, height=512, bg="white")
        self.canvas.pack(side=tk.LEFT)

        self.side_frame = tk.Frame(self.master)
        self.side_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.open_button = tk.Button(self.side_frame, text="Open", command=self.open_file)
        self.open_button.pack()

        self.save_button = tk.Button(self.side_frame, text="Save", command=self.save_rotated_image)
        self.save_button.pack()
        
        self.rotation_entry = tk.Entry(self.side_frame)
        self.rotation_entry.pack()
        
        self.rotate_button = tk.Button(self.side_frame, text="Rotate", command=self.rotate_by_entry)
        self.rotate_button.pack()

        self.position_button = tk.Button(self.side_frame, text="Track Position", command=self.toggle_position_tracking)
        self.position_button.pack()
        
        self.position_tracking = False
        self.clicked_marker = None
        
        self.position_label = tk.Label(self.side_frame, text="Position on canvas: x:  y: ")
        self.position_label.pack()

        self.image = None
        self.tk_image = None
        self.angle = 0

        self.coordinates = []  # 用於存儲葉片輪廓的位點
        self.contour_id = None  # 存储轮廓的ID

        # 添加加载葉片輪廓文件的选项
        self.select_points_button = tk.Button(self.side_frame, text="Select Points", command=self.toggle_select_points)
        self.select_points_button.pack()
        self.select_points_mode = False
        self.selected_points = []

        # 添加保存位点的按钮
        self.save_points_button = tk.Button(self.side_frame, text="Save Points", command=self.save_points_between_selected)
        self.save_points_button.pack()
        self.save_points_button.config(state=tk.DISABLED)  # 初始状态下按钮不可用

        # 轮廓点列表窗口
        self.contour_window = tk.Toplevel(self.master)
        self.contour_window.title("Contour Points")

        self.contour_list = tk.Listbox(self.contour_window, width=20)
        self.contour_list.pack(side=tk.LEFT, fill=tk.Y)

        self.contour_scroll = tk.Scrollbar(self.contour_window, orient=tk.VERTICAL)
        self.contour_scroll.config(command=self.contour_list.yview)
        self.contour_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.contour_list.config(yscrollcommand=self.contour_scroll.set)
        self.contour_list.bind("<<ListboxSelect>>", self.select_contour_point)

    def create_menu(self):
        menubar = tk.Menu(self.master)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_separator()
        
        remove_background_menu = tk.Menu(file_menu, tearoff=0) 
        remove_background_menu.add_command(label="Open Image", command=self.open_image)
        remove_background_menu.add_command(label="Open Directory", command=self.open_dir)
        file_menu.add_cascade(label="Remove Background", menu=remove_background_menu)

        # 添加加载葉片輪廓文件的选项
        file_menu.add_command(label="Load Contour", command=self.load_contour)

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
        self.canvas.delete("all")
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
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def save_rotated_image(self):
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filename:
            blank_image = Image.new("RGB", (512, 512), color="white")
            rotated_image = self.image.rotate(self.angle, resample=Image.BICUBIC, expand=True, fillcolor="white")
            blank_image.paste(rotated_image, (0, 0))
            blank_image.save(filename)

    def open_image(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png")])
        if file_paths:
            for file_path in file_paths:
                if not process_images(file_path):  
                    messagebox.showerror("Error", f"Failed to remove background for {file_path}")
                else:
                    messagebox.showinfo("Background Removed", f"Background removed successfully for {file_path}")

    def open_dir(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            if not process_images(folder_path):  
                messagebox.showerror("Error", f"Failed to remove background for images in {folder_path}")
            else:
                messagebox.showinfo("Background Removed", f"Background removed successfully for images in {folder_path}")

    def toggle_select_points(self):
        # 切换选择位点的模式
        self.select_points_mode = not self.select_points_mode
        if self.select_points_mode:
            self.select_points_button.config(text="Stop Selecting Points")
            self.canvas.bind("<Button-1>", self.select_point)
        else:
            self.select_points_button.config(text="Select Points")
            self.canvas.unbind("<Button-1>")
            self.canvas.delete("selected_point")
            if len(self.selected_points) == 2:
                self.save_points_button.config(state=tk.NORMAL)  # 当选定两个点后，保存按钮变为可用状态
            else:
                self.save_points_button.config(state=tk.DISABLED)  # 当未选定两个点时，保存按钮不可用

    def show_position(self, event):
        if self.coordinates:
            x, y = event.x, event.y
            if self.is_inside_contour(x, y):
                position_text = f"Position on canvas: x: {x} y: {y}"
                self.position_label.config(text=position_text)
                self.canvas.delete("current_position")
                self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", tags="current_position")
            else:
                self.position_label.config(text="Position on canvas: x:  y: ")
                self.canvas.delete("current_position")

    def is_inside_contour(self, x, y):
        if not self.coordinates:
            return False
        return self.canvas.find_overlapping(x, y, x, y) == (self.contour_id,)

    def load_contour(self):
        filename = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if filename:
            self.coordinates = self.load_coordinates(filename)
            self.draw_contour()
            self.update_contour_list()

    def load_coordinates(self, filename):
        coordinates = []
        with open(filename, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                # 解析字符串，提取数字
                x, y = map(int, row[0].strip("[]").split())
                coordinates.append((x, y))
        return coordinates

    def draw_contour(self):
        if self.coordinates:
            self.canvas.delete("contour")  # 清除之前的輪廓
            self.contour_id = self.canvas.create_line(self.coordinates, fill="green", width=2, tags="contour")  # 繪製新輪廓

    def update_contour_list(self):
        self.contour_list.delete(0, tk.END)
        for index, point in enumerate(self.coordinates, start=1):
            self.contour_list.insert(tk.END, f"Point {index}: ({point[0]}, {point[1]})")

    def select_contour_point(self, event):
        # 获取所选点的索引
        selected_index = self.contour_list.curselection()
        if selected_index:
            index = selected_index[0]
            self.canvas.delete("selected_point")
            x, y = self.coordinates[index]
            self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", tags="selected_point")

    def toggle_position_tracking(self):
        # 切换位置追踪功能
        self.position_tracking = not self.position_tracking
        if self.position_tracking:
            self.position_button.config(text="Stop Tracking Position")
            self.canvas.bind("<Motion>", self.show_position)
        else:
            self.position_button.config(text="Track Position")
            self.canvas.unbind("<Motion>")
            self.canvas.delete("current_position")
            self.position_label.config(text="Position on canvas: x:  y: ")

    def select_point(self, event):
        # 记录选定的点
        x, y = event.x, event.y
        self.selected_points.append((x, y))
        self.canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="blue", tags="selected_point")
        if len(self.selected_points) == 2:
            self.toggle_select_points()
            self.save_points_between_selected()
    
    def save_points_between_selected(self):
        if len(self.selected_points) == 2:
            # 提示用户选择保存文件的位置
            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if filename:
                with open(filename, "w", newline="") as file:
                    writer = csv.writer(file)
                    for point in self.coordinates:
                        if self.is_inside_selection(point):
                            writer.writerow(point)


    def show_contour_window(self):
        if self.contour_window:
            self.contour_window.destroy()
        self.contour_window = tk.Toplevel(self.master)
        self.contour_window.title("Contour Points")

        scrollbar = tk.Scrollbar(self.contour_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        listbox = tk.Listbox(self.contour_window, yscrollcommand=scrollbar.set)
        for point in self.coordinates:
            listbox.insert(tk.END, f"{point[0]}, {point[1]}")
        listbox.pack(side=tk.LEFT, fill=tk.BOTH)

        scrollbar.config(command=listbox.yview)

        # 添加 Entry 和 Add 按钮
        self.p1_entry = tk.Entry(self.contour_window)
        self.p1_entry.pack()

        self.p2_entry = tk.Entry(self.contour_window)
        self.p2_entry.pack()

        self.add_button = tk.Button(self.contour_window, text="Add", command=self.add_points_to_entries)
        self.add_button.pack()

        save_button = tk.Button(self.contour_window, text="Save Selected Points", command=self.save_selected_points)
        save_button.pack()

    def add_points_to_entries(self):
        selected_indices = self.contour_window.children["!listbox"].curselection()
        if len(selected_indices) == 1:
            point = self.coordinates[selected_indices[0]]
            if not self.p1_entry.get():
                self.p1_entry.insert(tk.END, f"{point[0]}, {point[1]}")
            elif not self.p2_entry.get():
                self.p2_entry.insert(tk.END, f"{point[0]}, {point[1]}")
            else:
                messagebox.showwarning("Warning", "Both entry fields are already filled.")

    def save_selected_points(self):
        p1 = self.p1_entry.get()
        p2 = self.p2_entry.get()
        if p1 and p2:
            p1_x, p1_y = map(int, p1.split(","))
            p2_x, p2_y = map(int, p2.split(","))
            min_x = min(p1_x, p2_x)
            max_x = max(p1_x, p2_x)
            min_y = min(p1_y, p2_y)
            max_y = max(p1_y, p2_y)

            filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if filename:
                with open(filename, "w", newline="") as file:
                    writer = csv.writer(file)
                    for point in self.coordinates:
                        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                            writer.writerow(point)


    def is_inside_selection(self, point):
        # 检查一个点是否在所选区域内
        x, y = point
        x1, y1 = self.selected_points[0]
        x2, y2 = self.selected_points[1]
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def main():
    root = tk.Tk()
    app = ImageRotator(root)
    root.mainloop()

if __name__ == "__main__":
    main()
