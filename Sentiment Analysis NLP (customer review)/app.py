from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.core.window import Window
from numpy import double

Window.clearcolor = '#E6DCD0'




from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re #regex function


class app(App):
    def build(self):
        self.window = GridLayout()
        self.window.cols = 1
       
        self.window.padding = 50
       
        
        #self.window.add_widget(Image(source='cl.png'))
        self.msg = Label(text="Customer Review",
                         font_size = 25,
                         color = '#302D2A')
        
        self.window.add_widget(self.msg)

        self.rating = Image(source="0.png")
       
        self.window.add_widget(self.rating)
        self.user = TextInput(font_size = 20,
                      size_hint_y = 1,
                      height = 50,
                      
                              
                              )
        self.window.add_widget(self.user)

        self.button = Button(text="Submit",size_hint_y = None, size_hint_x  = 5, height = 50)
       
        self.button.bind(on_press = self.callback)
       
        self.window.add_widget(self.button)
        return self.window


    def callback(self,instance):
                
        #Instantiate Model

        tokenzier = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
        model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

        #Encode and Calculate Sentiment

        tokens = tokenzier.encode(self.user.text,return_tensors='pt')
        print(tokens[0])

        result = model(tokens)

        result.logits
        nb = double(torch.argmax(result.logits))+1
        print(int(torch.argmax(result.logits))+1)
        print(result)

        self.msg.text = "Rating : "+str(nb)+" out of 5"

        if(nb == 5):
          self.rating.source = "5.png"
        elif(nb == 4):
            self.rating.source = "4.png"
        elif(nb== 3):
            self.rating.source = "3.png"
        elif(nb == 2):
            self.rating.source = "2.png"
        elif(nb == 1):
            self.rating.source = "1.png"
        elif(nb == 0):
            self.rating.source = "0.png"
       
    




if __name__ == "__main__":
    app().run()