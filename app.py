from flask import Flask, render_template, request, jsonify
import matplotlib.image as image
import collections
import pandas as pd
import pickle
import joblib

app = Flask(__name__)
classification_dict = {"0": "Non conforme", "1": "conforme"}
# Extraire les features de l'image
model = pickle.load(open("models/model_rf.pkl","rb"))
scaler = pickle.load(open("models/scaler.pkl","rb"))

# with open('model_rf.pickle','rb') as f:
#     model = pickle.load(f)
#
# with open('scaler.pickle','rb') as g:
#     scaler = pickle.load(g)

#model = joblib.load("models/model_rf.sav")
#scaler = joblib.load("models/scaler.sav")

def extraire_feature(img1):
    img = image.imread(img1)
    pixels_line_i = []
    for i in range(512):
        for j in range(512):
            pixels_line_i.append(img[i][j][0])
    counter = dict(collections.Counter(pixels_line_i))
    keys_values = counter.items()
    new_d = {str(key): [value] for key, value in keys_values}
#    for k in range(256):
#        if str(k) not in new_d.keys():
#           new_d[str(k)] = [0]
    dict_new = dict()
    for feature in range(256):
        if feature in list(counter.keys()):
            dict_new[feature] = [counter[feature]]
        else:
            dict_new[feature] = [0]
#    df = pd.DataFrame(new_d)
    df = pd.DataFrame(dict_new)
    X_std = scaler.transform(df)
    return df,X_std


@app.route("/", methods=['GET', 'POST'])
def main():

    return render_template('index.html')


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    print("ok")
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/"+img.filename
        img.save(img_path)
        x,x_std = extraire_feature(img_path)
        print("test",x_std.min(),x_std.max())
        p = model.predict(x_std)
        print("p",p)
        p0 = p[0]
        print("p1",p0)
        y = classification_dict[str(p0)]
        print("value of p0",y)
    return render_template('index.html', prediction=y, img_path=img_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
