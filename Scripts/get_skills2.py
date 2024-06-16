'''All functions for extracting skills from an input text and map them onto a predefined dictionary'''

# import necessary libraries
from dotenv import load_dotenv
import os
from openai import AzureOpenAI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Union
from fastapi import FastAPI
from embetter.text import SentenceEncoder
from neofuzz import Process
from gliner import GLiNER
import csv
import pickle
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat
from typing import List, Dict
import json
import pprint
import time
from configparser import ConfigParser
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)



config = ConfigParser()
config.read('./config.ini')

paths = dict(config['paths'])

mapper_config = dict(config['mapper_config'])
general_params = dict(config['general'])
domain = general_params['domain']
dashboard_params = dict(config['dashboard'])

if domain!= 'dashboard':
    domain_params = dict(config[domain])
    


load_dotenv()
openai_config = dict(config['openai_config'])
openai_config['endpoint'] = eval(openai_config['endpoint'])
openai_config['azure_api_key'] = eval(openai_config['azure_api_key'])
openai_config['direct_openai_api_key'] = eval(openai_config['direct_openai_api_key'])
openai_config['use_azure_openai'] = eval(openai_config['use_azure_openai'])


llm_config = dict(config['llm_config'])

gliner_config = dict(config['gliner_config'])


#domain_params = dict(config['domain_params'])

# Config variables
base_dir = paths['base_dir']
data_dir = paths['data_dir']


indexed_process = None
gliner_model = None

terms2cat = None # a global dict (term:['domain','domain_spec_category',generic_category]) to avoid reload

# internal variables

class TextInput(BaseModel):
    textEntered: str
    domain: str

#class GetmeJSON_all_skills(BaseModel):
#
#    hard_skills: List[str]
#    soft_skills: List[str]
    
class GetmeJSON_all_skills2(BaseModel):

    skills: List[str]
    
    
   

#schema_all_skills = GetmeJSON_all_skills.model_json_schema() # returns a dict like JSON schema
schema_all_skills2 = GetmeJSON_all_skills2.model_json_schema() # returns a dict like JSON schema

response_format = ResponseFormat(type="json_object")

#verbose = True

class ModelOutput(BaseModel):
    hardSkills: List[str]
    humanSkills: List[str]
    unmatchedSkills: List[str]
    
class ModelOutput2(BaseModel):
    skills: Dict
# Load environment variables from the .env file (if present)


# Access environment variables as if they came from the actual environment
#openai_config = {}
#openai_config['endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT')
#openai_config['api_key'] = os.getenv('AZURE_OPENAI_API_KEY')
#openai_config['api_version'] = "2024-02-01"
#openai_config['model'] = "gpt-4o"




# initialize Azure OpenAI client



# client = AzureOpenAI(
#   azure_endpoint = openai_config['endpoint'],
#   api_key = openai_config['api_key'],
#   api_version = openai_config['api_version'])



# helper functions
def create_client(openai_config:dict)->AzureOpenAI:
    '''Create an OpenAI client'''
    
    if openai_config['use_azure_openai']:
        client = AzureOpenAI(
            azure_endpoint = openai_config['endpoint'],
            api_key = openai_config['azure_api_key'],
            api_version = openai_config['api_version'])
    else:
        client = OpenAI(
            api_key=openai_config['direct_openai_api_key'])
    return client


def get_references2(domains :List[str] = ['dashboard','analytics'])->List[str]:
    '''unified_list = get_references2(domains=['dashboard','analytics'])'''
    
    global terms2cat 
    
    unified_list = []
    
    unified_list = list(terms2cat.keys())
    logging.info(f'{unified_list[:5]=}')
    
    return unified_list     
            
            
# def get_references(skill_type: str ='soft_skills', sources :List[str] = ['ESCO','Lightcast'])->List[str]:    
#     '''unified_list = get_references(skill_type='soft_skills', sources = ['ESCO','Lightcast'])'''
    
#     unified_list = []
#     for source in sources:
#         path_to_file = data_dir+skill_type+'/'+source+'_'+skill_type+'.csv'
        
#         with open(path_to_file, mode ='r') as file:
#             csvfile = csv.reader(file)
            
#             unified_list = list(csvfile)
#             unified_list = sum(unified_list, [])
           
#     return unified_list


def get_indexed_neofuzz_process2(mapper_config:dict = mapper_config)->Process:
    '''Returns an indexed neofuzz process object. If it does not exist, it will be created.'''
    
    
    global terms2cat
    
    references = json.loads(mapper_config['references'])
    logging.info('%s type of references',type(references))
    
    filename = '_'.join(references['domains'])+'_terms2cat.pkl'
    logging.info(f'{filename=}')
    with open(data_dir+filename, 'rb') as f:
        terms2cat = pickle.load(f)
    
    tmp = list(terms2cat.keys())
    logging.info(f'terms2cat: {tmp[0]}:{terms2cat[tmp[0]]}')
    
    
    embedding_model = mapper_config['embedding_model']
    
    model_name = '_'.join(references['domains'])+'_'+'indexed_process.pkl'
    path_to_model = mapper_config['path']+model_name
    
    logging.info('%s path to model',path_to_model)
    if os.path.isfile(path_to_model):
        logging.info('neofuzz models found!\n')
        
        with open(path_to_model, 'rb') as f:
            neofuzz_process = pickle.load(f)   
    else:
        logging.info('neofuzz models are being generated!\n')
        
        vectorizer = SentenceEncoder(embedding_model)
        neofuzz_process = Process(vectorizer, metric="cosine",n_jobs=-1)
        unified_references = get_references2(domains=references['domains'])
        neofuzz_process.index(unified_references)
        #neofuzz_process.to_disk(path_to_model)
        with open(path_to_model, 'wb') as f:
            pickle.dump(neofuzz_process, f) 
             
    return neofuzz_process
    
    
    

def skill_mapper2(phrases:List|str, mapper_config:dict = mapper_config, paths:dict = paths)->ModelOutput2:
    
    global indexed_process
    global terms2cat
    
    
    
    if indexed_process is None:
        indexed_process = get_indexed_neofuzz_process2(mapper_config=mapper_config)  
    
    tmp = list(terms2cat.keys())
    logging.info(f'terms2cat: {terms2cat[tmp[0]]}')
    
    categories = json.loads(mapper_config['categories'])
    #category_mapping = json.loads(mapper_config['category_mapping'])
    mapped_skills = {}
    unmatched_skills = []
    
    domain_list = [ ]
    for d in ['dashboard','analytics']:
        mapped_skills[d] = {}
        for category in categories[d]:
            mapped_skills[d][category] = []
    
    cat_list = []
    generic_cat_list = []
    terms = []
    
    logging.info('%s type(phrases)',type(phrases))
    
    domain_preference = bool(mapper_config['domain_preference'])
    threshold = float(mapper_config['mapping_similarity_threshold'])
    score_diff = float(mapper_config['maximum_score_diff'])
    
    if isinstance(phrases, str):
        logging.info('Phrases is a dict') #llm returns a srt (json object), gliner returns a list
        tmp = json.loads(phrases)
        logging.info(f'llm output: {tmp}')
        phrases = tmp['skills']
    
    
    for phrase in phrases:
        results = indexed_process.extract(phrase, limit=2) #neofuzz returns the 2 most similar terms from the references
        logging.info(f'{phrase=} {results=}')    
        
        logging.info(f'{terms2cat[results[0][0]]=}')    
        if (results[0][1]>threshold) & (results[1][1]<=threshold): #First match only scores high enough
            logging.info('First match only scores high enough')
            terms.append(results[0][0])
            domain_list.append(terms2cat[results[0][0]][0])
            cat_list.append(terms2cat[results[0][0]][1])
            generic_cat_list.append(terms2cat[results[0][0]][2])
        elif results[1][1]>threshold: #Both matches score high
            logging.info('Both matches score high')
            if (domain_preference==True) & (results[0][1]-results[1][1]<score_diff) & (terms2cat[results[0][0]][1]=='dashboard') & (terms2cat[results[1][0]][1]!= 'dashboard') :
                terms.append(results[1][0])
                domain_list.append(terms2cat[results[1][0]][0])
                cat_list.append(terms2cat[results[1][0]][1])
                generic_cat_list.append(terms2cat[results[1][0]][2])
            else:
                terms.append(results[0][0])
                domain_list.append(terms2cat[results[0][0]][0])
                cat_list.append(terms2cat[results[0][0]][1])
                generic_cat_list.append(terms2cat[results[0][0]][2])
        else:
            unmatched_skills.append(phrase)
    
    logging.info(f'{terms[:5]=} {domain_list[:5]} {cat_list[:5]=}')                
    #domains = list(set(domain_list))
    #for domain in domains:
    #    mapped_skills[domain] = {}
    #    for cat in categories[domain]:
    #        mapped_skills[domain][cat] = []
   
    
    #remove duplicate mappings
    idx_list = [idx for idx, val in enumerate(terms) if val in terms[:idx]]
    
    reduced_terms = list(filter(lambda x: terms.index(x) not in idx_list, terms))
    reduced_domain_list = list(filter(lambda x: domain_list.index(x) not in idx_list, domain_list))
    reduced_cat_list = list(filter(lambda x: cat_list.index(x) not in idx_list, cat_list))
    reduced_generic_cat_list = list(filter(lambda x: generic_cat_list.index(x) not in idx_list, generic_cat_list))
    
    logging.info(f'{reduced_terms[:5]=} {reduced_domain_list[:5]=} {reduced_cat_list[:5]=} {reduced_generic_cat_list[:5]}')
    
        
    for i in range(len(reduced_terms)):
        if reduced_domain_list[i]=='dashboard':
            mapped_skills[reduced_domain_list[i]][reduced_cat_list[i]].append(reduced_terms[i])
        else:
            mapped_skills[reduced_domain_list[i]][reduced_cat_list[i]].append(reduced_terms[i])
            mapped_skills['dashboard'][reduced_generic_cat_list[i]].append(reduced_terms[i])  
    
    return mapped_skills




def extract_with_gliner(text:str, gliner_config:dict)->List[str]:    
    '''Return one list of skills extracted by GLiNER.'''
    
    
    path_to_gliner_model = config['gliner_config']['path'] 
    gliner_threshold = config['gliner_config']['threshold']
    gliner_model = GLiNER.from_pretrained(path_to_gliner_model, local_files_only=True) 
    
    predicted_skills = []
    # Perform entity prediction
    entities = gliner_model.predict_entities(text, ["skill"], threshold=gliner_threshold)

    # Display predicted entities and their labels
    for entity in entities:
        #print(entity["text"], "=>", entity["label"])
        predicted_skills.append(entity["text"])
        
    return predicted_skills


def create_prompt_message2(text:str,llm_config:dict,domain:str,domain_params:dict|None)->str:
    '''Return the prompt message for the LLM.'''
    
    #single_prompt = llm_config['single_prompt']
    
    gpt_assistant_prompt = '''You are a professional recruiter and you can extract the required skills from a 
    resume, a job posting or from a course material.'''
    
    if domain == 'dashboard':
        gpt_user_prompt = f'''The definitions of skills are the following: {dashboard_params['skill_def']}. Based on these definitions identify and extract the 
        specific skills from the below text: {text}. Rely only on the text provided! If you cannot identify skills, return an empty list!
        Return the response in JSON format!'''
    else:
        gpt_user_prompt =f'''The definitions of skills are the following: {dashboard_params['skill_def']}. In addition, let us focus on the following, 
        domain-specific definition:{domain_params['skill_def']}. Based on these definitions identify and extract the 
        generic and the domain specific skills from the text below: {text}. Rely only on the text provided! If you cannot identify skills, return an empty list!
        Return the response in JSON format!'''
    
    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt}]
    
    return message


def create_prompt_message3(text:str,llm_config:dict,domain:str,domain_params:dict|None)->str:
    '''Return the prompt message for the LLM.
    message = create_prompt_message(text,llm_config,domain,domain_params)'''
    
    #single_prompt = llm_config['single_prompt']
    
    gpt_assistant_prompt = '''You are a professional recruiter and you can identify and extract job related skills from a given text, be it a 
    resume, a job posting or a course material.'''
    
    if domain == 'dashboard':
        gpt_user_prompt = f'''The definitions of skills are the following: {dashboard_params['skill_def']}. Based on these definitions identify and extract the 
        specific skills from the below text: {text}. Rely only on the text provided! If you cannot identify skills, return an empty list!
        Return the response in JSON format!'''
    else:
        gpt_user_prompt =f'''First, focus on the domain-specific skills for which the  definition is the following: {domain_params['skill_def']}. Based on this definition, 
        identify and extract the domain-specific skills from the text below: {text}. 
        Now consider again the entire text and identify and extract the more generic skills. For that the following definition may help: {dashboard_params['skill_def']}.
        Rely only on the text provided! If you cannot identify skills, return an empty list!
        Return the response in JSON format!'''
    
    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt}]
    
    return message




def call_llm(client:AzureOpenAI,message:str,schema,response_format,model):
  """output = get_skills(client, message, schema,response_format,model='gpt-4o')
  IN:
  client: Azure OpenAI client object
  message: list of dicts
  schema: JSON scheme
  response_format:
  model: valid Azure OpenAI model name

  OUT:
  output: json object

  """
  response = client.chat.completions.create(
      model=model,
      response_format=response_format,
      messages=message,
      temperature = 0.2,
      frequency_penalty =0.0,
      functions=[
          {
            "name": "get_answer_for_user_query",
            "description": "Get user answer.",
            "parameters": schema
          }
      ],
      function_call={"name": "get_answer_for_user_query"}
  )
  json_object = response.choices[0].message.function_call.arguments
  if json_object:
    logging.info(f'full response: {json.loads(json_object)}')
  else:
    logging.info('empty response from LLM!!!')
    json_object = json.dumps({'skills': [ ]}) 

  return json_object


def extract_with_llm2(text,domain,llm_config,schema=schema_all_skills2,response_format=response_format,openai_config=openai_config):    
    
    
    message = create_prompt_message2(text, llm_config,domain=domain,domain_params=domain_params)
    
    logging.info('message: %s',message)
    
    client = create_client(openai_config)
    predicted_skills = call_llm(client,message,schema,response_format,openai_config['model'])
    return predicted_skills



    
def get_mapped_skills2(textinput:TextInput,method:str='llm',gliner_config = gliner_config,llm_config = llm_config, openai_config=openai_config,mapper_config = mapper_config,\
    schema=schema_all_skills2,response_format=response_format)->ModelOutput:   
    
    '''mapped_skills = get_mapped_skills(text, method,llm_config,openai_config,mapper_config)''' 
    
    text = textinput.textEntered
    domain = textinput.domain
    
    if method=='gliner':
        predicted_skills = extract_with_gliner(text, gliner_config)    
    elif method=='llm':
        predicted_skills = extract_with_llm2(text,domain,llm_config,schema=schema_all_skills2,response_format=response_format,openai_config=openai_config)   
    else:
        raise ValueError('Invalid method. Please use either "gliner" or "llm".')
    
      
    logging.info('predicted skills: %s',predicted_skills) 
    mapped_skills = skill_mapper2(predicted_skills,mapper_config)   
        
    
    return mapped_skills





if __name__ == '__main__':
    
    # ml_models = {}

    # @asynccontextmanager
    # async def ml_lifespan_manager(app: FastAPI):
    #     # load the ml model and prediction logic
    #     ml_models["get_mapped_skills"] = get_mapped_skills
    #     yield
    #     # release the resources + cleanup
    #     ml_models.clear()

    # app = FastAPI(lifespan=ml_lifespan_manager)

    # @app.get("/")
    # async def root():
    #     return {"message": "I do not like empty pages..."}


    # @app.post("/predict")
    # async def predict(text_input: TextInput):
    #     # return {"extracted_skills":ml_models["get_mapped_skills"](text_input,method='llm',hard_skills_definition=hard_skills_definition,
    #     #                                                    soft_skills_definition=soft_skills_definition,
    #     #                                                    path_to_gliner_model=path_to_gliner_model,
    #     #                                                    openai_config=openai_config,
    #     #                                                    threshold=mapping_similarity_threshold)}
    #     return {"extracted_skills":ml_models["get_mapped_skills"](text_input,method='llm')}
        
        
    
    
    test_input = '''We are looking to students who are enthusiastic about driving a sustainable, 
    net-zero future, with a focus on developing strategies to reduce carbon emissions and optimize 
    operational energy consumption of infrastructure, encompassing its construction, utilization and end of life phases, 
    through effective carbon management and decarbonisation plans.Conduct research and analysis on carbon emissions inventories, 
    reduction strategies, and regulatory requirements. Managing own workload to meet project deadlines.Reviewing project details 
    and developing targets for sustainability.Reviewing documentation provided by project teams, assess this against the project’s 
    targets and provide feedback or follow-up. Communicating sustainability requirements to internal and external team members. 
    Producing reports providing advice and guidance to clients and project teams. Attending site visits to review progress against 
    sustainability targets. Collaborate with multidisciplinary teams to develop carbon reduction targets and action plans for clients 
    across different sectors.Strong academic background with coursework or experience in carbon accounting, greenhouse gas emissions, 
    or related fields. A passion for innovation and problem solving. Humility, self-motivation and enthusiasm. Excellent communication, 
    presentation and writing skills. An understanding of the construction industry and built environment.
    '''
        
    test_input2 = '''I am a web programmer with knowledge on node.js, express, mongodb, and react. I have experience in building web applications.'''
    
    #text_input = TextInput(domain = domain,textEntered = '''We are looking to students who are enthusiastic about driving a sustainable, net-zero future, with a focus on developing strategies to reduce carbon emissions and optimize operational energy consumption of infrastructure, encompassing its construction, utilization and end of life phases, through effective carbon management and decarbonisation plans.Conduct research and analysis on carbon emissions inventories, reduction strategies, and regulatory requirements. Managing own workload to meet project deadlines.Reviewing project details and developing targets for sustainability.Reviewing documentation provided by project teams, assess this against the project’s targets and provide feedback or follow-up. Communicating sustainability requirements to internal and external team members. Producing reports providing advice and guidance to clients and project teams. Attending site visits to review progress against sustainability targets. Collaborate with multidisciplinary teams to develop carbon reduction targets and action plans for clients across different sectors.Strong academic background with coursework or experience in carbon accounting, greenhouse gas emissions, or related fields. A passion for innovation and problem solving. Humility, self-motivation and enthusiasm. Excellent communication, presentation and writing skills. An understanding of the construction industry and built environment.
    #''')
    
    text_input = TextInput(domain = domain,textEntered =test_input2)
    

    
    start_time = time.perf_counter () 
    mapped_skills = get_mapped_skills2( 
        text_input,
        method='llm',)
    
    end_time = time.perf_counter ()
    print(end_time - start_time, "seconds")
    
    """ start_time = time.perf_counter () 
    mapped_skills = get_mapped_skills2( 
        text_input,
        method='llm')
    
    end_time = time.perf_counter ()
    print(end_time - start_time, "seconds") """
    
    # #mapped_skills = get_skills(text_input,method='llm', path_to_gliner_model=path_to_gliner_model, client=client, openai_config=openai_config,threshold = mapping_similarity_threshold)
    pprint.pprint(mapped_skills)
    
    # #extracted_skills = extract_skills()
    
    