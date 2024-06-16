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



load_dotenv()
openai_config = dict(config['openai_config'])
openai_config['endpoint'] = eval(openai_config['endpoint'])
openai_config['api_key'] = eval(openai_config['api_key'])


llm_config = dict(config['llm_config'])

gliner_config = dict(config['gliner_config'])




# Config variables
#base_dir = '/home/gabor/Work/MAGAN/Abodoo/Codes2/skill_extractor_app/'
base_dir = paths['base_dir']
data_dir = paths['data_dir']
#neofuzz_model_dir = base_dir+ 'Models/Neofuzz/'
#gliner_model_dir = base_dir+ 'Models/GLiNER/'
#gliner_model = 'large_v21_general'

#path_to_gliner_model = gliner_model_dir+gliner_model
#gliner_threshold = 0.1





indexed_process_soft_skills = None
indexed_process_hard_skills = None
gliner_model = None

# internal variables

class TextInput(BaseModel):
    textEntered: str

class GetmeJSON_all_skills(BaseModel):

    hard_skills: List[str]
    soft_skills: List[str]
   

schema_all_skills = GetmeJSON_all_skills.model_json_schema() # returns a dict like JSON schema
response_format = ResponseFormat(type="json_object")

#verbose = True

class ModelOutput(BaseModel):
    hardSkills: List[str]
    humanSkills: List[str]
    unmatchedSkills: List[str]

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


def get_references(skill_type: str ='soft_skills', sources :List[str] = ['ESCO','Lightcast'])->List[str]:    
    '''unified_list = get_references(skill_type='soft_skills', sources = ['ESCO','Lightcast'])'''
    
    unified_list = []
    for source in sources:
        path_to_file = data_dir+skill_type+'/'+source+'_'+skill_type+'.csv'
        
        with open(path_to_file, mode ='r') as file:
            csvfile = csv.reader(file)
            
            unified_list = list(csvfile)
            unified_list = sum(unified_list, [])
           
    return unified_list


#def get_indexed_neofuzz_process(skill_type: str = 'soft_skills',references :dict = references, embedding_model:str = 'all-mpnet-base-v2')->Process:
def get_indexed_neofuzz_process(skill_type: str = 'soft_skills',mapper_config:dict = mapper_config)->Process:    
    '''Returns an indexed neofuzz process object. If it does not exist, it will be created.'''
    
    references = json.loads(mapper_config['references'])
    logging.info('%s type of references',type(references))
   
    embedding_model = mapper_config['embedding_model']
    
    
    model_name ='_'.join(references[skill_type])+'_'+skill_type+'_indexed_process.pkl'
    #path_to_model = neofuzz_model_dir+model_name
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
        unified_references = get_references(skill_type=skill_type,sources = references[skill_type])
        neofuzz_process.index(unified_references)
        #neofuzz_process.to_disk(path_to_model)
        with open(path_to_model, 'wb') as f:
            pickle.dump(neofuzz_process, f) 
             
    return neofuzz_process
        
        

#Gliner model 
# def get_gliner_model(path_to_gliner_model:str)->GLiNER:
#     gliner_model = GLiNER.from_pretrained(path_to_gliner_model, local_files_only=True)    
#     return gliner_model


#def skill_mapper(phrases:List|str, references:dict = references, embedding_model:str = embedding_model, threshold:float = mapping_similarity_threshold)->ModelOutput:
def skill_mapper(phrases:List|str, mapper_config)->ModelOutput:    
    '''mapped_skills = skill_mapper(phrases, references, embedding_model,threshold)
    
    For each phrase in the list of phrases, it will try to semantically match it with the soft skills and the hard skills. The input may be a list 
    or  a dict of two separate lists.'''
    
    global indexed_process_hard_skills
    global indexed_process_soft_skills
    
    if indexed_process_hard_skills is None:
        indexed_process_hard_skills = get_indexed_neofuzz_process(skill_type='hard_skills',mapper_config=mapper_config)       

    if indexed_process_soft_skills is None:
        indexed_process_soft_skills = get_indexed_neofuzz_process(skill_type='soft_skills',mapper_config=mapper_config)
    
    
    hard_skills = []
    soft_skills = []
    unmatched_skills = []
    
    logging.info('%s type(phrases)',type(phrases))
    
    
    threshold = float(mapper_config['mapping_similarity_threshold'])

    if isinstance(phrases, list):
    
        for phrase in phrases:
            results_hardskill = indexed_process_hard_skills.extract(phrase, limit=1)
            results_softskill = indexed_process_soft_skills.extract(phrase, limit=1) 
            if (results_hardskill[0][1]>results_softskill[0][1]) & (results_hardskill[0][1]>threshold):
                hard_skills.append(results_hardskill[0][0])
            elif (results_softskill[0][1]>results_hardskill[0][1]) & (results_softskill[0][1]>threshold):
                soft_skills.append(results_hardskill[0][0])    
            else:
                unmatched_skills.append(phrase)

        mapped_skills = ModelOutput(hardSkills=hard_skills,humanSkills=soft_skills,unmatchedSkills=unmatched_skills)
    
    if isinstance(phrases, str):
        
        logging.info('Phrases is a dict')
        phrases = json.loads(phrases)
        for skill_type, skill_list in phrases.items():
            if skill_type == 'hard_skills':
                for phrase in skill_list:
                    results_hardskill = indexed_process_hard_skills.extract(phrase, limit=1)
                    if results_hardskill[0][1]>threshold:
                        hard_skills.append(results_hardskill[0][0])
                    else:
                        unmatched_skills.append(phrase)
                        
            if skill_type == 'soft_skills':
                for phrase in skill_list:
                    results_softskill = indexed_process_soft_skills.extract(phrase, limit=1)
                    if results_softskill[0][1]>threshold: 
                        soft_skills.append(results_softskill[0][0])
                    else:
                        unmatched_skills.append(phrase)
                        
        
        logging.info('hard skills: %s',hard_skills)
        logging.info('soft skills: %s',soft_skills)
        logging.info('unmatched skills: %s',unmatched_skills)
        
        
        mapped_skills = ModelOutput(hardSkills=hard_skills,humanSkills=soft_skills,unmatchedSkills=unmatched_skills)

    return mapped_skills




#def extract_with_gliner(text:str, path_to_gliner_model:str = path_to_gliner_model, gliner_threshold:float = gliner_threshold)->List[str]:
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



#def create_prompt_message (text:str, hard_skills_definition:str, soft_skills_definition:str)->str:
def create_prompt_message (text:str, llm_config:dict)->str:
    '''Return the prompt message for the LLM.'''
    
    hard_skills_definition = llm_config['hard_skills_definition']
    soft_skills_definition = llm_config['soft_skills_definition']
    
    gpt_assistant_prompt = "You are a professional recruiter and you can extract the required skills from a resume, a job posting or from a course material."
    #gpt_user_prompt = f'''Based on the definitions of {soft_skills_definition} and {hard_skills_definition} identify 
    #and extract the mentioned skills of the below profile text: {text}. Return the response in JSON format!'''
    gpt_user_prompt = f'''Identify and extract the required hard skills and soft skills from the below profile text: {text}. Rely only on the text provided!
    If you cannot identify hard skills or soft skills, return an empty list for the corresponding skill type!
    Return the response in JSON format!'''
  
    message=[{"role": "assistant", "content": gpt_assistant_prompt}, {"role": "user", "content": gpt_user_prompt}]
    return message



def call_llm(client:AzureOpenAI,message:str,schema,verbose,response_format,model):
  """output = get_skills(client, message, schema,verbose,response_format,model='gpt-4o')
  IN:
  client: Azure OpenAI client object
  message: list of dicts
  schema: JSON scheme
  verbose: Bool
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
  if verbose:
    pprint.pp(json.loads(json_object))

  return json_object


#def extract_with_llm2(text,hard_skills_definition,soft_skills_definition,schema,verbose,response_format,threshold=0.5,openai_config=openai_config):
def extract_with_llm2(text,llm_config,schema=schema_all_skills,verbose=True,response_format=response_format,openai_config=openai_config):    
    
    
    message = create_prompt_message(text, llm_config)
    
    logging.info('message: %s',message)
    
    client = create_client(openai_config)
    predicted_skills = call_llm(client,message,schema,verbose,response_format,openai_config['model'])
    return predicted_skills
    

# def extract_with_llm(text,azureclient=client, openai_config=openai_config):
#     '''Return both the hard and soft skills extracted by the LLM. The return format is a JSON object with the following structure: 'hard_skills:[]' and 'soft_skills:[]'.'''
    
#     gpt_assistant_prompt = "You are a professional recruiter and you can extract the required skills from a resume, a job posting or from a course material."   
    
#     messages = [
#             {"role": "system", "content": "You are a professional HR recruiter who can extract the required skills from a resume, a job posting or from a course material."},
#             {"role": "user","content": f'''Extract the hard skills and soft skills identified from the following text: {text}. Return the skills in a JSON format with the 
#             following structure: {{hard_skills: ['skill1','skill2',...], soft_skills: list['skill1','skill2',...]}}. If you do not find any skills, 
#             return an empty list. Return the JSON object only, do not add any extra text.'''  
#             } 
#           ]
    
    
#     response = azureclient.chat.completions.create(
#     model = openai_config['model'],  # model = "deployment_name".
#     messages=messages,
#     )
    
#     predicted_skills = response.choices[0].message.content
#     predicted_skills = json.loads(predicted_skills)
#     print(predicted_skills)
    
#     return predicted_skills





# def get_mapped_skills(
#         text:str,
#         method:str ='llm',
#         hard_skills_definition:str = hard_skills_definition, 
#         soft_skills_definition:str = soft_skills_definition, 
#         schema:GetmeJSON_all_skills = schema_all_skills, 
#         verbose: bool = verbose,
#         path_to_gliner_model:str=path_to_gliner_model, 
#         openai_config:AzureOpenAI =openai_config,
#         embedding_model:str = embedding_model,
#         response_format:ResponseFormat = response_format,
#         threshold:float = mapping_similarity_threshold)->ModelOutput:
    
def get_mapped_skills(textinput:TextInput,method:str='llm',gliner_config = gliner_config,llm_config = llm_config, openai_config=openai_config,mapper_config = mapper_config,\
    schema=schema_all_skills,verbose=True,response_format=response_format)->ModelOutput:   
    
    '''mapped_skills = get_mapped_skills(text, method,llm_config,openai_config,mapper_config)''' 
    
    text = textinput.textEntered
    
    if method=='gliner':
        predicted_skills = extract_with_gliner(text, gliner_config)    
    elif method=='llm':
        predicted_skills = extract_with_llm2(text,llm_config,schema=schema_all_skills,verbose=verbose,response_format=response_format,openai_config=openai_config)   
    else:
        raise ValueError('Invalid method. Please use either "gliner" or "llm".')
    
      
    logging.info('predicted skills: %s',predicted_skills) 
    mapped_skills = skill_mapper(predicted_skills,mapper_config)   
        
    
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
        
        
        
    
    
    text_input = TextInput(textEntered = '''Job Title: Senior Lead Technician/Programmer
    Company: Security 101
    Location: San Antonio, TX
    Job Summary:
    Security 101 - San Antonio is now recruiting a commercial electronic security programmer with experience in configuring, testing, and 
    commissioning enterprise access control and video management systems.
    Salary and Benefits:
    This is a Full-time position with an annual salary from $60,000 to $70,00, exempt, commensurate with skills, product knowledge and experience. 
    Our benefits include medical, dental, vision, prescription coverage, paid holidays, unlimited PTO and more. We will provide a vehicle, laptop, and smart phone.
    The Right Candidate:
    Must have at least 4 years' career experience installing, servicing, and programming commercial access control, 
    IP-based video systems and intrusion alarm systems.
    Experience with access control panel programming.
    The ability to build a video or access control server from the ground up.
    Familiarity/experience with alarm panel configuration and programing.
    Understanding of SQL and databases a plus.
    Valid TX driver's license and clean MVR.
    Current certification in one or more of the following:
    Access Control:
    Genetec, Software House, Avigilon Unity, Avigilon Alta and Brivo.
    VMS:
    Milestone, Genetec, Avigilon Unity, Avigilon Alta.
    Security 101 will invest in training for the right candidate.
    Responsibilities:
    Troubleshoot wide area/segmented networks.
    Work directly with end users on how to operate systems.
    Work directly with IT departments to assist with needed network topography for security equipment.
    Familiarity with SIP IP phone systems a bonus
    Managing quality control and providing fanatical customer service
    Taking responsibility for and tracking project progress.
    Education:
    High School (or GED) minimum requirement
    Electronics training from a Technical School or Military training in electronics or communications (preferred)
    About Security 101:
    Security 101 is a provider of integrated electronic security solutions to a diversified set of commercial customers across multiple end markets, 
    including healthcare, education, financial, and government, among others. Security 101 delivers a full-service offering of electronic 
    security services and products including the design, installation, and maintenance of access control, video surveillance, intrusion detection,
    and visitor management solutions. Founded in 2005 and based in West Palm Beach, FL, Security 101 has 52 locations in the U.S. 
    For more information, please visit www.security101.com.
    Security 101
    is a DFWP and EOE organization with a team-oriented work environment.
    ''')
    

    
    start_time = time.perf_counter () 
    mapped_skills = get_mapped_skills( 
        text_input,
        method='llm',)
    
    end_time = time.perf_counter ()
    print(end_time - start_time, "seconds")
    
    start_time = time.perf_counter () 
    mapped_skills = get_mapped_skills( 
        text_input,
        method='llm')
    
    end_time = time.perf_counter ()
    print(end_time - start_time, "seconds")
    
    # #mapped_skills = get_skills(text_input,method='llm', path_to_gliner_model=path_to_gliner_model, client=client, openai_config=openai_config,threshold = mapping_similarity_threshold)
    # pprint.pprint(f'{mapped_skills}')
    
    # #extracted_skills = extract_skills()
    
    