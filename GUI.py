# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 13:18:34 2023

@author: shais
"""


import numpy as np
import torch
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

updated_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def select_image():
    # Function to handle image selection
    file_path = filedialog.askopenfilename()
    fp.set(file_path)
    if file_path:
        # Load the image
        image = Image.open(file_path)

        # Resize the image if needed
        image = image.resize((600, 600))

        # Convert the image to a Tkinter-compatible format
        tk_image = ImageTk.PhotoImage(image)
        updated_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        dict1 = {"0": 5, "1": 20, "2": 30, "3": 40, "4": 10, "5": 20, "6": 20, "7": 20, "8": 20, "9": 100}
        for i in range(len(updated_values)):
            box_labels[i].config(text=str(int(dict1[str(i)])))
            no_labels[i].config(text=str(updated_values[i]))
            
            

        # Update the image label
        bill_label.configure(text="Total Bill:\n Rs{:.2f}".format(0))
        image_label.configure(image=tk_image)
        image_label.image = tk_image

def calculate_bill():
    model = YOLO("D://shayan//Food//IndianFood10//FoodDetect//Food10s//weights//best.pt")
    res = model(fp.get())
    res1 = res[0].boxes
    img = cv2.imread(fp.get())
    dict1 = {"0": 5, "1": 20, "2": 30, "3": 40, "4": 10, "5": 20, "6": 20, "7": 20, "8": 20, "9": 100}
    dict2 = {"0": "alooparatha", "1": "rasgulla", "2": "biryani", "3": "chickentikka", "4": "palakpaneer",
             "5": "poha", "6": "khichdi", "7": "omelette", "8": "plainrice", "9": "chapati"}
    updated_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    price = 0
    for box in res1:
        x = int(box.xyxy[0][0].tolist())
        y = int(box.xyxy[0][1].tolist())
        x1 = int(box.xyxy[0][2].tolist())
        y1 = int(box.xyxy[0][3].tolist())
        cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)
        c1 = dict2[str(int(box.cls.tolist()[0]))]
        cv2.putText(img, c1, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        updated_values[int(box.cls.tolist()[0])] += 1
        price += dict1[str(int(box.cls.tolist()[0]))]

    for i in range(len(updated_values)):
        box_labels[i].config(text=str(int(dict1[str(i)])))
        no_labels[i].config(text=str(updated_values[i]))
       
    img1 = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img1 = img1.resize((600, 600))
    tk1_image = ImageTk.PhotoImage(img1)
    image_label.configure(image=tk1_image)
    image_label.image = tk1_image

    total_bill = price
    bill_label.configure(text="Total Bill:\n Rs{:.2f}".format(total_bill))


# Create the main window
root = tk.Tk()
root.title("Food Billing System")
root.geometry("1300x700")
root.configure(bg="#f2f2f2")

fp = tk.StringVar()

# =============================================================================
# lf = tk.Frame(root, bg="#f2f2f2")
# lf.pack(side=tk.LEFT, padx=100,pady=10)
# =============================================================================

# =============================================================================
# rf = tk.Frame(root, bg="#f2f2f2")
# rf.pack(side=tk.RIGHT, padx=100)
# =============================================================================

# Create a button to select the image
select_button = tk.Button(root, text="Select Image", command=select_image, bg="#4CAF50", fg="white", padx=10, pady=5)
select_button.grid(row=0,column=0)

# =============================================================================
# text_label=tk.Label(text="Please select the desired image", font=("Arial", 12, "italic"), bg="#f2f2f2")
# text_label.pack()
# =============================================================================

# Create a label to display the selected image
demo_image = Image.open("demo.jpg")
demo_image = demo_image.resize((600, 600))
tk_demo_image = ImageTk.PhotoImage(demo_image)

image_label = tk.Label(root, image=tk_demo_image, padx=10, pady=0)
image_label.grid(row=1,column=0,rowspan=12,padx=50)

food_classes = ["Aloo Paratha", "Rasgulla", "Biryani", "Chicken Tikka", "Palak Paneer",
                "Poha", "Khichdi", "Omelette", "Plain Rice", "Chapati"]

food_classes_price = {"Aloo Paratha":5, "Rasgulla":20, "Biryani":30, "Chicken Tikka":40, "Palak Paneer":10,
                "Poha":20, "Khichdi":20, "Omelette":20, "Plain Rice":20, "Chapati":100}
food_label = tk.Label(root, text=" Food Classes",font=("Arial", 12, "bold"), padx=10, pady=10)
food_label.grid(row=0,column=1)

priceof = tk.Label(root, text=" Price",font=("Arial", 12, "bold"), padx=10, pady=10)
priceof.grid(row=0,column=2)

nop = tk.Label(root, text="No. of items",font=("Arial", 12,"bold"), padx=10, pady=10)
nop.grid(row=0,column=3)

c=1
r=1
no_labels=[]
box_labels = []
for food_class in food_classes:
    # Create a label for the heading
    label = tk.Label(root, text=food_class, font=("Arial", 10, "bold"), bg="#f2f2f2")
    label.grid(row=r,column=c,padx=50)

    # Create a label for the box value
    box_label = tk.Label(root, text="0", font=("Arial", 10,"bold"), bg="#f2f2f2")
    box_label.grid(row=r,column=c+1,padx=50)

    box_labels.append(box_label)
    
    no_label = tk.Label(root, text="0", font=("Arial", 10,"bold"), bg="#f2f2f2")
    no_label.grid(row=r,column=c+2,padx=50)

    no_labels.append(no_label)
    
    r+=1


bill_button = tk.Button(root, text="Calculate Bill", command=calculate_bill, bg="#4CAF50", fg="white", padx=10, pady=5)
bill_button.grid(row=11,column=2,padx=50)

bill_label = tk.Label(root, text="Total Bill:\n Rs 0.00", font=("Arial", 10, "bold"), padx=10, pady=10, bg="#f2f2f2")
bill_label.grid(row=12,column=2,padx=50)


# =============================================================================
# some=tk.Label(root,text=" edjbwqhdhwqihwq ")
# some.grid(row=13,column=0,columnspan=4,pady=0)
# =============================================================================


# =============================================================================
# root.grid_propagate(False)
# =============================================================================
# Run the GUI application
root.mainloop()