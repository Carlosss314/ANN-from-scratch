import numpy as np
import matplotlib.pyplot as plt

from tkinter import *
import PIL
from PIL import Image, ImageDraw

def draw_number():
    def paint(event):
        x1, y1 = (event.x + 1), (event.y + 1)
        x2, y2 = (event.x - 1), (event.y - 1)
        cv.create_rectangle(x1, y1, x2, y2, fill="black", width=10)  # On tkinter Canvas
        draw.rectangle((x1/10, y1/10, x2/10, y2/10), fill="white", width=1)  # On PIL Canvas

    def save():
        image1.save("image.png")

    app = Tk()

    cv = Canvas(app, width=280, height=280, bg='white')
    cv.pack()

    image1 = Image.new("L", (28, 28), (0))
    draw = ImageDraw.Draw(image1)

    cv.bind("<B1-Motion>", paint)

    button=Button(text="save", command=save)
    button.pack()

    app.mainloop()

def read_image():
    image = PIL.Image.open("image.png")
    image_array = np.array(image)
    image_array = image_array.reshape(784, 1) / 255
    return image_array

def load_weights_and_bias():
    w1 = np.load("weights_and_bias/w1.npy")
    b1 = np.load("weights_and_bias/b1.npy")
    w2 = np.load("weights_and_bias/w2.npy")
    b2 = np.load("weights_and_bias/b2.npy")
    return w1, b1, w2, b2


#model
def ReLU(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_prop(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def get_predictions(a2):
    prediction = np.argmax(a2, axis=0)
    prediction = prediction.reshape(prediction.shape[0], 1)
    return prediction


draw_number()
image_array = read_image()
w1, b1, w2, b2 = load_weights_and_bias()
z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, image_array)
prediction = get_predictions(a2)



plt.figure()
plt.imshow(image_array.reshape(28, 28), cmap="Greys")
plt.title(f"prediction: {prediction}")
plt.show()
