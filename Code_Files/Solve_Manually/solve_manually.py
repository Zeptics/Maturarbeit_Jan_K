import os
import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk, ImageDraw
from tkinter import font as tkFont  # Import font to customize text size
import time

root = tk.Tk()

def run_main_menu(root, img_folder, img_names_list):
    class MainMenu:
        def __init__(self, window):
            self.window = window

            self.window.rowconfigure(0)
            self.window.rowconfigure(1)
            self.window.columnconfigure(0)

            self.grid_size = (4, 4)  # 4x4 grid
            self.img_counter = -1

            self.window.title("Captcha Selection")

            # Load image and get dimensions
            self.img, self.img_tk = self.load_image(os.path.join(img_folder, img_names_list[self.img_counter]),
                                     (600, 600))  # Resizing to 400x400 for display

            # Canvas to display image
            self.canvas = Canvas(self.window, width=self.img.width, height=self.img.height)
            self.canvas.grid(row=1, column=0, padx=5)

            # Add the button to switch through the images
            self.next_button_font = tkFont.Font(size=20, weight='bold')
            self.next_button = tk.Button(text='Next', background='#71aee0', font=self.next_button_font, command=self.change_image)
            # self.next_button.grid(row=2, column=0, pady=10)

            # Add counter to show progress
            self.counter_font = tkFont.Font(size=16)
            self.counter_label = tk.Label(text=f'{self.img_counter + 1}/{number_of_images}', font=self.counter_font)
            # self.counter_label.grid(row=0, column=0, pady=10)

            # Add the button to start
            self.start_button_font = tkFont.Font(size=20, weight='bold')
            self.start_button = tk.Button(text='Start', background='#71aee0', font=self.start_button_font, command=self.start_time)
            self.start_button.grid(row=2, column=0, pady=10)

            # Display the image on the canvas
            # self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)

            # Set to keep track of selected grid areas
            self.selections = set()

            self.start_time = 0

            # Bind mouse click event to toggle selection
            # self.canvas.bind("<Button-1>", lambda event: self.toggle_selection(event))

            root.mainloop()

        def start_time(self):
            self.counter_label.grid(row=0, column=0, pady=10)
            self.change_image()
            self.start_button.grid_remove()
            self.next_button.grid(row=2, column=0, pady=10)
            self.canvas.bind("<Button-1>", lambda event: self.toggle_selection(event))
            self.start_time = time.time()



        def save_image_matrix(self):
            intersection_matrix = [[False, False, False, False],
                                         [False, False, False, False],
                                         [False, False, False, False],
                                         [False, False, False, False]]
            for item in self.selections:
                intersection_matrix[item[0]][item[1]] = True

            with open(os.path.join(os.getcwd(), 'matrix_output', f'{img_names_list[self.img_counter][:-4]}.txt'), 'w') as file:
                file.write(f'{intersection_matrix}')

        def change_image(self):
            self.save_image_matrix()
            self.img_counter += 1
            if self.img_counter < number_of_images:
                self.img, self.img_tk = self.load_image(os.path.join(img_folder, img_names_list[self.img_counter]),
                                                        (600, 600))

                # Clear the selections when changing the image
                self.selections.clear()

                # Update the existing canvas with the new image, don't recreate it
                self.canvas.create_image(0, 0, anchor="nw", image=self.img_tk)
                self.counter_label = tk.Label(text=f'{self.img_counter + 1}/{number_of_images}', font=self.counter_font)
                self.counter_label.grid(row=0, column=0)

                self.add_grid_to_img()

            else:
                elapsed_time = time.time() - self.start_time
                with open(os.path.join(os.getcwd(), 'matrix_output', 'statistics.txt'), 'w') as file:
                    file.write(f'Total time: {elapsed_time} seconds\n')
                    file.write(f'Time per image: {elapsed_time/number_of_images} seconds')
                label_font = tkFont.Font(size=26)
                done_label = tk.Label(text='Du bist fertig, danke!', font=label_font)
                done_label.grid(row=1, column=0)

        def add_grid_to_img(self):
            temp_img = self.img.copy()
            draw = ImageDraw.Draw(temp_img)
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()

            for vertical_line_number in [width/4, width/4*2, width/4*3]:
                draw.line([vertical_line_number, 0, vertical_line_number, height], width=3)

            for horizontal_line_number in [height/4, height/4*2, height/4*3]:
                draw.line([0, horizontal_line_number, width, horizontal_line_number], width=3)

            img_tk_overlay = ImageTk.PhotoImage(temp_img)
            self.canvas.img_tk_overlay = img_tk_overlay
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk_overlay)

        def draw_selections(self):
            temp_img = self.img.copy()
            draw = ImageDraw.Draw(temp_img)

            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            col_width = width // self.grid_size[0]
            row_height = height // self.grid_size[1]

            for (row, col) in self.selections:
                x0 = col * col_width
                y0 = row * row_height
                x1 = x0 + col_width
                y1 = y0 + row_height
                draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 255, 128))  # White overlay with transparency

            # To keep the grid after selecting squares
            for vertical_line_number in [width/4, width/4*2, width/4*3]:
                draw.line([vertical_line_number, 0, vertical_line_number, height], width=3)
            for horizontal_line_number in [height/4, height/4*2, height/4*3]:
                draw.line([0, horizontal_line_number, width, horizontal_line_number], width=3)

            img_tk_overlay = ImageTk.PhotoImage(temp_img)
            self.canvas.img_tk_overlay = img_tk_overlay
            self.canvas.create_image(0, 0, anchor="nw", image=img_tk_overlay)

        def toggle_selection(self, event):
            width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
            col_width = width // self.grid_size[0]
            row_height = height // self.grid_size[1]

            col = event.x // col_width
            row = event.y // row_height

            grid_index = (row, col)

            if grid_index in self.selections:
                self.selections.remove(grid_index)
            else:
                self.selections.add(grid_index)

            self.draw_selections()

        def load_image(self, path, size):
            img = Image.open(path)
            img = img.resize(size)
            return img, ImageTk.PhotoImage(img)

    MainMenu(root)
    root.mainloop()


images_folder = "test_images"
images_names_list = os.listdir(images_folder)
number_of_images = len(os.listdir(images_folder))

os.makedirs(os.path.join(os.getcwd(), 'matrix_output'), exist_ok=True)

run_main_menu(root, images_folder, images_names_list)