import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# variables globales
ancho_global = 0
alto_global = 0
img_source = -1

def redimensionar_con_aspecto(imagen, max_ancho, max_alto):
    global ancho_global, alto_global
    ancho, alto = imagen.size
    ratio = min(max_ancho/ancho, max_alto/alto)
    nuevo_ancho = int(ancho * ratio)
    nuevo_alto = int(alto * ratio)

    ancho_global, alto_global = nuevo_ancho, nuevo_alto
    return imagen.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)

def cargar_imagen():
    global img_source 
    img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']

    ruta = filedialog.askopenfilename(
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.gif *.bmp")]
    )
    if not ruta:
        return
    elif os.path.isfile(ruta):
        _, ext = os.path.splitext(ruta)
        if ext in img_ext_list:
            img = Image.open(ruta)
            img = redimensionar_con_aspecto(img, 1200, 1000)
            tk_img = ImageTk.PhotoImage(img)
            etq_original.config(image=tk_img)   
            etq_original.image = tk_img
            img_source = ruta
        else:
            print(f'File extension {ext} is not supported.')
            sys.exit(0)
    else:
        print(f'Tiene que cargar una imagen')
        sys.exit(0)

def procesar_imagen():
    global img_source, ancho_global, alto_global
    model_path = "best.pt"
    
    # Load the model into memory and get labemap
    model = YOLO(model_path, task='detect')
    labels = model.names

    # Set bounding box colors (using the Tableu 10 color scheme)
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    frame = cv2.imread(img_source)
    
    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes


    # Initialize variable for basic object counting example
    object_count = 0

    # Go through each detection and get bbox coords, confidence, and class
    for i in range(len(detections)):

        # Get bounding box coordinates
        # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
        xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
        xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
        print(xyxy)
        xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

        # Get bounding box class ID and name
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        print(classname)

        # Get bounding box confidence
        conf = detections[i].conf.item()
        print(conf)

        # Draw box if confidence threshold is high enough
        if conf > 0.1:

            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 10)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.5, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 2) # Draw label text

            # Basic example: count the number of objects in the image
            object_count = object_count + 1
        
        # Display detection results
        cv2.putText(frame, f'Number of objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2) # Draw total number of detected objects

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        pil_img = pil_img.resize((ancho_global, alto_global), Image.Resampling.LANCZOS)
        tk_img2 = ImageTk.PhotoImage(pil_img)

        etq_procesada.config(image=tk_img2)
        etq_procesada.image = tk_img2


    if (not os.path.exists(model_path)):
        print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
        sys.exit(0)


# Ventana y grilla
ventana = tk.Tk()
ventana.title("Visor de Imágenes")
for col in (0, 2):
    ventana.columnconfigure(col, weight=1)
ventana.columnconfigure(1, weight=3)
ventana.rowconfigure(0, weight=1)
ventana.rowconfigure(1, weight=0)

# Etiqueta de la imagen
etq_original = tk.Label(ventana)
etq_original.grid(row=0, column=0, padx=10, pady=10)

etq_procesada = tk.Label(ventana)
etq_procesada.grid(row=0, column=1, padx=10, pady=10)

# Frame de botones
frame_botones = tk.Frame(ventana)
frame_botones.grid(
    row=1,
    column=1,
    pady=10,
    sticky="nw",
    padx=20    # positivo, para desplazar desde el borde izquierdo
)

# Botones
boton_cargar = tk.Button(frame_botones, text="Cargar Imagen", command=cargar_imagen)
boton_cargar.pack(side="left", padx=5)

boton_procesar = tk.Button(frame_botones, text="Procesar", command=procesar_imagen)
boton_procesar.pack(side="left", padx=5)

ventana.mainloop()
