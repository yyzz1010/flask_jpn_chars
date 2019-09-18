from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
from keras.applications import vgg16
from keras.preprocessing import image
import imageio

app = Flask(__name__)

loaded_model = load_model('my_model.h5')
loaded_model._make_predict_function()

@app.route('/')
def home():
    return render_template("home.html")

def ClassPredictor(file):
    result = loaded_model.predict(file)
    return result[0]

def process_image(img):
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    result = ClassPredictor(img)
    return result

@app.route('/result', methods=['POST'])
def result():
    prediction = ''
    if request.method == 'POST':
        file = request.files['file']
        print('request.files:', request.files)
        img = np.array(image.load_img(file, target_size=(64,64)))
        result = process_image(img)
        print('result from model:', result)
        print(type(result))
        predicted_class_indices = np.argmax(result)
        print('predicted_class_indices:', predicted_class_indices)

        if predicted_class_indices == 0:
            prediction = 'Hiragana'
        else:
            prediction = 'Katakana'
        print(prediction)

        return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
