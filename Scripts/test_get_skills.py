#import sys
#sys.path.insert(0, '/home/gabor/Work/MAGAN/Abodoo/Codes2/skill_extractor_app/Scripts/')

#from get_skills import get_mapped_skills
import get_skills
import time
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List,Union
from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat
import pprint


# Config variables
base_dir = '/home/gabor/Work/MAGAN/Abodoo/Codes2/skill_extractor_app/'
data_dir = base_dir+ 'Data/'
neofuzz_model_dir = base_dir+ 'Models/Neofuzz/'
gliner_model_dir = base_dir+ 'Models/GLiNER/'
gliner_model = 'large_v21_general'

path_to_gliner_model = gliner_model_dir+gliner_model
gliner_threshold = 0.1

mapping_similarity_threshold = 65
embedding_model = 'all-mpnet-base-v2'

soft_skills_definition = "Soft skills include attributes and personality traits that help employees effectively interact with others and succeed in the workplace. \
                          These skills can include social graces, communication abilities, personal habits, cognitive or emotional empathy, time management, teamwork and leadesrship traits. \
                          The most important attributes are problem-solving, effective communication, self-direction,drive, adaptability, teamwork, dependability, conflict resolution,\
                          research, creativity, work ethic and integrity."
                          
hard_skills_definition = "Hard skills, or technical skills, are learned through education or hands-on experience. \
                          Examples of hard skills include security complience, blueprint reading, programming, computer skills, data visualization or cost-benefit analysis. These are concrete, measurable abilities that are often specific to a job."



indexed_process_soft_skills = None
indexed_process_hard_skills = None
gliner_model = None

class GetmeJSON_all_skills(BaseModel):

    hard_skills: List[str]
    soft_skills: List[str]
   

schema_all_skills = GetmeJSON_all_skills.model_json_schema() # returns a dict like JSON schema
response_format = ResponseFormat(type="json_object")

verbose = True

class ModelOutput(BaseModel):
    hardSkills: List[str]
    humanSkills: List[str]
    unmatchedSkills: List[str]

# Load environment variables from the .env file (if present)
load_dotenv()

# Access environment variables as if they came from the actual environment
openai_config = {}
openai_config['endpoint'] = os.getenv('AZURE_OPENAI_ENDPOINT')
openai_config['api_key'] = os.getenv('AZURE_OPENAI_API_KEY')
openai_config['api_version'] = "2024-02-01"
openai_config['model'] = "gpt-4o"


if __name__ == '__main__':

    text_input = '''Job Title: Senior Lead Technician/Programmer
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
        Security 101 is a provider of integrated electronic security solutions to a diversified set of commercial customers across multiple end markets, including healthcare, education, financial, and government, among others. Security 101 delivers a full-service offering of electronic security services and products including the design, installation, and maintenance of access control, video surveillance, intrusion detection, and visitor management solutions. Founded in 2005 and based in West Palm Beach, FL, Security 101 has 52 locations in the U.S. For more information, please visit www.security101.com.
        Security 101
        is a DFWP and EOE organization with a team-oriented work environment.
        '''
    

    
    start_time = time.perf_counter () 
    mapped_skills = get_skills.get_mapped_skills( 
                                    text_input,
                                    method='llm',
                                    hard_skills_definition=hard_skills_definition,
                                    soft_skills_definition=soft_skills_definition,
                                    path_to_gliner_model=path_to_gliner_model,
                                    openai_config=openai_config,
                                    threshold=mapping_similarity_threshold)
    
    end_time = time.perf_counter ()
    print(end_time - start_time, "seconds")
    
    start_time = time.perf_counter () 
    mapped_skills = get_skills.get_mapped_skills( 
                                    text_input,
                                    method='llm',
                                    hard_skills_definition=hard_skills_definition,
                                    soft_skills_definition=soft_skills_definition,
                                    path_to_gliner_model=path_to_gliner_model,
                                    openai_config=openai_config,
                                    threshold=mapping_similarity_threshold)
    
    end_time = time.perf_counter ()
    print(end_time - start_time, "seconds")
    
    #mapped_skills = get_skills(text_input,method='llm', path_to_gliner_model=path_to_gliner_model, client=client, openai_config=openai_config,threshold = mapping_similarity_threshold)
    pprint.pprint(f'{mapped_skills}')