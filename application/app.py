from flask import Flask, render_template, request
from keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

keras.backend.set_session(session)


app = Flask(__name__)

m =  load_model('C:/Users/dchen/Desktop/flask-application/defect.h5',custom_objects={'GlorotUniform': glorot_uniform()})
m._make_predict_function()

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        if request.files["image"]:
            try:
                with session.as_default():
                    with session.graph.as_default():
                        img = Image.open(request.files['image'].stream)
                        img = img.resize((64,64))
                        img_predict = np.expand_dims(img, axis=0)
                        result = m.predict(img_predict)
                        if result[0][0] == 1:
                            print("OK")
                            return render_template('success.html')
                        else:
                            print("Defect")
                            #return render_template('index.html')
                            return render_template('defect.html')
                            #add route called defect piece

            except Exception as ex:
                #log.log('Seatbelt Prediction Error', ex, ex.__traceback__.tb_lineno)
                print(ex.__traceback__.tb_lineno)
        else:
            return render_template('error.html')
    else:
            return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=False)


