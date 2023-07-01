import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request

app=Flask(__name__)
model=load_model("animalWeights.h5")
@app.route('/')
def index():
    return render_template("log.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        img=image.load_img(filepath,target_size=(200,200))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=np.argmax(model.predict(x),axis=1)

        if pred[0]==0:
            text= "Apple:\nCalories: 52\nWater: 86% \nProtein: 0.3 grams \nCarbs: 13.8 grams\nSugar: 10.4 grams\nFiber: 2.4 grams\nFat: 0.2 grams"
        elif pred[0]==1:
            text= "Banana \nCalories: 89\nWater: 72%\nProtein: 1.1 grams\nCarbs: 22.8 grams\nSugar: 12.2 grams\nFiber: 2.6 grams\nFat: 0.3 grams"
        elif pred[0]==2:
            text="Orange\nCalories: 66\nWater: 86%\nProtein: 1.3 grams\nCarbs: 14.8 grams\nSugar: 12 grams\nFiber: 2.8 grams\nFat: 0.2 grams"

        elif pred[0]==3:
            text="Pineapple\nCalories: 83\nVitamin : 88% of daily value\nProtein: 1 gram\nCarbs: 21.6 grams\nSugar: 10 grams\nFiber: 2.3 grams\nFat: 0 grams"

        elif pred[0]==4:
            text="Watermelon\nCalories: 30\nWater: 91%\nProtein: 0.6 grams\nCarbs: 7.6 grams\nSugar: 6.2 grams\nFiber: 0.4 grams\nFat: 0.2 grams"

      #  index=['Apples','Banana','Orange','Pineapple','Watermelon']
       # text="The Classified fruit is : " +str(index[pred[0]])
    return text

if __name__=='__main__':
    app.run()
