#!/usr/bin/env python
# coding: utf-8

# In[2]:


# this is where we create the web app using flask


# In[ ]:


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# In[3]:



app = Flask(__name__)    #creating web app
model = pickle.load(open('model.pkl', 'rb'))   #reading our model file

@app.route('/')               #these 3 lines are basically the homepage of the webapp. the homepage codes are in html page.
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])    # this is where the features are passed to the model to predict.
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]  # 'request' will take all the data entered in the text field of the website api and stored in int_features.
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Attrition rate will be  {}'.format(output))  #this goes into the {{prediction_text}} in html file


if __name__ == "__main__":
    app.run(debug=True)
    


# In[ ]:





# In[ ]:





# In[ ]:




