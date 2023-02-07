import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import transforms


from model import Model
from dataset import Dataset

TEST_DATA_DIR = 'ERA_Ticket_CAPTCHA_Dataset/testset/'

class CaptchaRecognizer:
    def __init__(self):
        self.model = Model()
        self.model.load_state_dict(torch.load('model_state_dict.pth'))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model.to(self.device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(self.rpt)
        ])
        self.charset = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def rpt(self, x):
        return x.repeat(1, 1, 1)

    def _preprocess_img(self, img: cv2.Mat) -> cv2.Mat:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        img = np.expand_dims(img, axis=2)
        img = cv2.resize(img, (90, 25))
        return self.transform(img)

    def recognize(self, img: cv2.Mat):
        img = self._preprocess_img(img)
        with torch.no_grad():
            inputs = torch.tensor(img, dtype=torch.float32)
            inputs = inputs.unsqueeze(0)
            inputs = Variable(inputs.to(self.device))
            outputs = self.model(inputs)
            outputs = outputs.view(-1, 36)
            outputs = nn.functional.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            outputs = outputs.view(-1, 4)
        
        return ''.join([self.charset[i] for i in outputs[0]])

if __name__ == '__main__':
    recognizer = CaptchaRecognizer()

    CAPTCHA_URL = 'https://web085004.adm.ncyu.edu.tw/NewSite/Captcha.ashx'
    HEADER = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.120 Safari/537.36',
    }

    import requests

    # show the captcha image and the predicted text in gui
    import tkinter as tk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title('Captcha Recognizer')
    root.geometry('300x300')

    label = tk.Label(root, image=None)
    label.pack()

    text = tk.Label(root, text='', font=('Arial', 32))
    text.pack()

    def next_captcha():
        response = requests.get(url=CAPTCHA_URL, headers=HEADER)
        captcha = response.content
        captcha = cv2.imdecode(np.frombuffer(captcha, np.uint8), cv2.IMREAD_COLOR)
        captcha_text = recognizer.recognize(captcha)
        img = ImageTk.PhotoImage(Image.fromarray(captcha))
        label.configure(image=img, width=150, height=50)
        label.image = img
        text.configure(text='predict: '+captcha_text)

    btn = tk.Button(root, text='next_captcha', command=next_captcha, font=('Arial', 32))
    btn.pack()

    root.mainloop()



    