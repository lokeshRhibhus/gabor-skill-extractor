'''Simple API for get_skills.py using FastAPI'''

from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
import os
import get_skills2
from get_skills2 import get_mapped_skills2
from pydantic import BaseModel
import configparser
from dotenv import load_dotenv
import json
import regex
 
 
config = configparser.ConfigParser()
config.read('./config.ini')

paths = dict(config['paths'])

mapper_config = dict(config['mapper_config'])



load_dotenv()
openai_config = dict(config['openai_config'])
openai_config['endpoint'] = eval(openai_config['endpoint'])
openai_config['api_key'] = eval(openai_config['api_key'])


llm_config = dict(config['llm_config'])

gliner_config = dict(config['gliner_config'])



class TextInput(BaseModel):
    textEntered: str
    domain:str

class ModelOutput(BaseModel):
    hardSkills: list
    humanSkills: list
    unmatchedSkills: list    



def remove_control_characters(text):
    return regex.sub(r'\p{C}', '', text)


ml_models = {}

@asynccontextmanager
async def ml_lifespan_manager(app: FastAPI):
    # load the ml model and prediction logic
    ml_models["get_mapped_skills"] = get_mapped_skills2
    yield
    # release the resources + cleanup
    ml_models.clear()

app = FastAPI(lifespan=ml_lifespan_manager)

@app.get("/")
async def root():
    return {"message": "I do not like empty pages..."}


@app.post("/predict/")
async def predict(text_input: TextInput):
        print("TextInput")
        #text_input = jsonable_encoder(text_input)
        #text_input.textEntered = jsonable_encoder(text_input.textEntered)
        #text_input.textEntered = remove_control_characters(text_input.textEntered)
        #text_input.textEntered = json.dumps(text_input.textEntered)
        #print(text_input)
        return {"extracted_skills":ml_models["get_mapped_skills"](text_input,method='llm')}