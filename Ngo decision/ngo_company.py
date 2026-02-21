from fastapi import FastAPI,Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from train_model import pca_performed

import pandas as pd
import pickle,ast

app = FastAPI()

with open("models/clustering_model.pkl",'rb') as file:
    clustering_model = pickle.load(file)

dataset = pd.read_csv('datasets/Country-data.csv')
dataset['net_exports'] = dataset['exports']-dataset['imports']
dataset = dataset.drop(['imports','exports','income'],axis=1)
with open("countries_name_list.txt",'r') as file:
    countries_names=ast.literal_eval(file.read())


template = Jinja2Templates(directory='templates')



@app.get("/",response_class=HTMLResponse)
def ngo_company_page(request:Request):
    
    return template.TemplateResponse('ngo_company.html',{'request':request,'countries':countries_names,'title':'NGO Company'})

@app.post("/ngo_company_data")
async def ngo_company_data(request:Request):
    data= await request.json()
    columns =['country','child_mort','health','inflation','life_expec','total_fer','gdpp','net_exports']
    dataframe = pd.DataFrame([data] , columns=columns)
    new_dataset = pd.concat([dataset,dataframe],axis=0).reset_index(drop=True)

    piplines=pca_performed()
    result=piplines.fit_predict(new_dataset)


    return {'data':int(result[-1])}