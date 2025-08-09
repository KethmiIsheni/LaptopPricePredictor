from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

def prediction(lst):
    filename = 'model/predictor.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    feature_names = [
        'Ram', 'Weight', 'Touchscreen', 'Ips',
        'Company_Acer', 'Company_Apple', 'Company_Asus', 'Company_Dell', 'Company_HP', 'Company_Lenovo',
        'Company_MSI', 'Company_Toshiba', 'Company_other',
        'TypeName_2 in 1 Convertible', 'TypeName_Gaming', 'TypeName_Netbook', 'TypeName_Notebook',
        'TypeName_Ultrabook', 'TypeName_Workstation',
        'OpSys_Linux', 'OpSys_Mac', 'OpSys_Other', 'OpSys_Windows',
        'cpu_name_AMD', 'cpu_name_Intel Core i3', 'cpu_name_Intel Core i5', 'cpu_name_Intel Core i7', 'cpu_name_Other',
        'gpu_name_AMD', 'gpu_name_ARM', 'gpu_name_Intel', 'gpu_name_Nvidia'
    ]
    X = pd.DataFrame([lst], columns=feature_names)
    pred_value = model.predict(X)
    return pred_value

@app.route('/', methods=['POST', 'GET'])
def index():
    pred_value=0
    if request.method == 'POST':
        ram = request.form.get('ram')
        weight = request.form.get('weight')
        company = request.form['company']
        typename = request.form['typename']
        opsys = request.form['opsys']
        cpu = request.form['cpuname']
        gpu = request.form['gpuname']
        touchscreen = request.form.getlist('touchscreen')
        ips = request.form.getlist('ips')

        feature_list = []
        # Numeric features
        feature_list.append(int(ram))   # Ram
        feature_list.append(float(weight))  # Weight
        feature_list.append(1 if 'touchscreen' in touchscreen else 0)  # Touchscreen
        feature_list.append(1 if 'ips' in ips else 0)  # Ips

        # One-hot encoding for Company
        company_list = ['Acer', 'Apple', 'Asus', 'Dell', 'HP', 'Lenovo', 'MSI', 'Toshiba', 'other']
        for c in company_list:
            feature_list.append(1 if company == c else 0)

        # One-hot encoding for TypeName
        typename_list = ['2 in 1 Convertible', 'Gaming', 'Netbook', 'Notebook', 'Ultrabook', 'Workstation']
        for t in typename_list:
            feature_list.append(1 if typename == t else 0)

        # One-hot encoding for OpSys
        opsys_list = ['Linux', 'Mac', 'Other', 'Windows']
        for o in opsys_list:
            feature_list.append(1 if opsys == o else 0)

        # One-hot encoding for CPU
        cpu_list = ['AMD', 'Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Other']
        for c in cpu_list:
            feature_list.append(1 if cpu == c else 0)

        # One-hot encoding for GPU
        gpu_list = ['AMD', 'ARM', 'Intel', 'Nvidia']
        for g in gpu_list:
            feature_list.append(1 if gpu == g else 0)

        # Now feature_list should have exactly 31 items
        pred_value = prediction(feature_list)* 1.79
        pred_value = round(pred_value[0], 2)  # Round to 2 decimal places
        print("Predicted value:", pred_value) 
        
    return render_template("index.html", pred_value=pred_value)
       
  

if __name__ == '__main__':
    #app.run(debug=True)
    app = Flask(__name__)