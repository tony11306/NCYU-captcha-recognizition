import flask
import requests
import datetime
import csv
from predict import CaptchaRecognizer
import cv2

app = flask.Flask(__name__, static_folder='static')
app.config["DEBUG"] = True
recognizer = CaptchaRecognizer()

CAPTCHA_SOURCE = 'https://web085004.adm.ncyu.edu.tw/NewSite/Captcha.ashx?d=eORHeRaQ3Q4k62cH'
cnt = 0

@app.route('/', methods=['GET'])
def home():

    tmp_img = requests.get(CAPTCHA_SOURCE)
    # store the image
    with open('static/captcha.png', 'wb') as f:
        f.write(tmp_img.content)

    img = cv2.imread('static/captcha.png')
    prediction = recognizer.recognize(img)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"labeled images count:{cnt}<br/><form action='/check' method='POST'><label for='captcha'><img src='static/captcha.png?{current_time}' /></label><input type='text' value='{prediction}' name='captcha' id='captcha' oninput='this.value = this.value.toUpperCase()' autofocus/><input type='submit' value='Submit'/></form>"

@app.route('/check', methods=['POST'])
def check():
    captcha = flask.request.form['captcha']
    
    
    # if captcha is all number, then direct home
    if captcha.isdigit():
        return flask.redirect(flask.url_for('home'))

    # write to csv
    csv_writer.writerow([captcha, captcha])
    # copy image to trainset
    with open('static/captcha.png', 'rb') as f:
        img = f.read()
    with open(f'NCYU_Captcha_Dataset/trainset/{captcha}.png', 'wb') as f:
        f.write(img)
    global cnt
    cnt += 1
    return flask.redirect(flask.url_for('home'))

if __name__ == '__main__':
    label_csv_path = 'NCYU_Captcha_Dataset/trainset/label.csv'
    f = open(label_csv_path, 'a')
    csv_writer = csv.writer(f)
    app.run()
    f.close()