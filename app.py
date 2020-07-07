# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:07:06 2020

@author: Win7
"""
from flask import Flask,render_template,url_for,request
app = Flask(__name__,template_folder='templates')
import pickle
grad=pickle.load(open('grad_boosting.pkl','rb'))

#grad.predict([[2,324,12,34,8,14,0,2,5,1,11,6]])
@app.route('/')
def flight_details():
   return render_template('flight.html')
   #return grad.predict([[2,324,12,34,8,14,0,2,5,1,11,6]])
#
@app.route('/predict',methods=["POST"])
def predict():
    Total_stop= int(request.form["Total_stop"])
    Duration= int(request.form["Duration"])
    Departure_hr=int(request.form["Departure_hr"])
    Departure_min=int(request.form["Departure_min"])
    Arrival_hr=int(request.form["Arrival_hr"])
    Arrival_min=int(request.form["Arrival_min"])
    Source=int(request.form["Source"])
    Destin=int(request.form["Destin"])
    add_info=int(request.form["extra facility"])
    Airline= int(request.form["Airline"])
    date=int(request.form["doj"].split('-')[2])
    month=int(request.form["doj"].split('-')[1])
    year=int(request.form["doj"].split('-')[0])
    
    pred=grad.predict([[Total_stop,Duration,Departure_hr,Departure_min,Arrival_hr,Arrival_min,Source,Destin,add_info,Airline,date,month,year]])
    opt=round(pred[0],2)
    return render_template('output.html',prediction_value='Flight cost is {}'.format(opt))
    #return Total_stop,Duration,Departure_hr,Departure_min,Arrival_hr,Arrival_min,Source,Destin,add_info,Airline,date,month,year
        
    
if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
#@app.route('/blog/<int:postID>')
#def show_blog(postID):
#   return 'Blog Number %d' % postID
#
#@app.route('/paragraph/')
#def para_graph():
#    return render_template('para.html')

