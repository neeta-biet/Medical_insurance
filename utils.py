import pickle 
import json
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")





class InsuranceMedical():
    def __init__(self,age,sex,bmi,children,smoker,region):
        self.age=age
        self.sex=sex
        self.bmi=bmi
        self.children=children
        self.smoker=smoker
        self.region="region_"+region
        
    def load_models(self):
        with open("Linear model.pkl", "rb") as f:
            self.model=pickle.load(f)
        
        with open("Project_data.json", "r") as f:
            self.json_data=json.load(f)
            
    def predicted_charges(self):
        self.load_models()
        region_index = list(self.json_data["columns"]).index(self.region)
        test_array = np.zeros(len(self.json_data["columns"]))
        test_array[0] = self.age
        test_array[1] = self.json_data['sex'][self.sex]
        test_array[2] = self.bmi
        test_array[3] = self.children
        test_array[4] = self.json_data['smoker'][self.smoker]
        test_array[region_index] = 1
        
        print("test_array is : \n",test_array)
        
        charges=self.model.predict([test_array])[0]
        
        return charges
    
if __name__=="__main__":
    age = 19.0
    sex = "female"
    bmi = 27.9
    children = 0.0
    smoker = "no"
    region = "southwest"
    
    med_ins=InsuranceMedical(age,sex,bmi,children,smoker,region)
    charges=med_ins.predicted_charges()
    print("Predicted Medical Insurance Charges is :", charges, "/- Rs. Only")
    
        
        
    
        
        
        
        
        
        