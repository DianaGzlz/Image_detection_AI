from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

imagenes = []
nombres = []

for i in os.listdir("imagenes_prueba/"): #cambiar por el directorio donde se encuentran las imag
    img = Image.open("imagenes_prueba/" + i) #cambiar por el directorio donde se encuentran las imag
    img = img.resize((256, 256), Image.LANCZOS) #cambiar por el tamaño de las imagenes
    img = np.array(img) #cambiar por el tamaño de las imagenes
    imagenes.append(img/255) #cambiar por el tamaño de las imagenes
    nombres.append(i) #guarda el nombre de la imagen
imagenes = np.array(imagenes) #convertir a array de numpy

model0 = load_model("modelos/model0.h5") #cargar el modelo original
model1 = load_model("modelos/model1.h5") #cargar el modelo modificado
etiquetas = ['airplane', 'car', 'cat', 'dog', 'flower', 'fruit', 'motorbike', 'person']

for i in imagenes: #por cada imagen en el conjunto de imagenes
    i = i.reshape(1, 256, 256, 3) #darle la forma correcta a la
    result = etiquetas[np.argmax(model0.predict(i))] #predecir la imagen con el modelo original
    result1 = etiquetas[np.argmax(model1.predict(i))] #predecir la imagen con el modelo modificado 
    print("Imagen: ", nombres[imagenes.tolist().index(i.tolist()[0])]) #imprimir el nombre de la imagen
    print("Predicción modelo 0: ", result) #imprimir la predicción del modelo original
    print("Predicción modelo 1: ", result1) #imprimir la predicción del modelo modificado




        
    

