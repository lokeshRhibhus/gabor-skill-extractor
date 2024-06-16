import get_skills2
from get_skills2 import get_mapped_skills2
from pydantic import BaseModel
import csv
from pathlib import Path
import json
import os
import jsonlines
import pandas as pd 

class TextInput(BaseModel):
    textEntered: str
    domain: str

domain = "grentech"
def batch_skill_extraction(file_paths):
    '''all_mapped_skills = batch_skill_extraction(file_paths)'''
    
    all_mapped_skills = {}
    for file_name in os.listdir(file_paths):
        
        f = os.path.join(file_paths, file_name)
        print(f)
        with open(f, "r") as file:
            content = json.load(file)
        text_input = TextInput(textEntered = content,domain=domain)
        mapped_skills = get_mapped_skills2(text_input, method='llm')
        print(mapped_skills)
        print(type(mapped_skills))
        all_mapped_skills[file_name] = mapped_skills
    return all_mapped_skills
        

def extract_from_csv(path_to_file,path_to_output,domain):
    
    all_mapped_skills = []
    counter = 0
    with jsonlines.open(path_to_output, mode='a') as writer:
        with open(path_to_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if counter>72:
                    x =row['Text']
                    x =" ".join(word for word in x.split() if not word.startswith("\\"))
                    text_input = TextInput(textEntered = x,domain=domain)
                    print(text_input.textEntered)
                    mapped_skills = get_mapped_skills2(text_input, method='llm')
               
                    all_mapped_skills.append(mapped_skills)
               
                    writer.write(mapped_skills)
                counter += 1    
                print(counter)
            
                # if counter == 3:
                #      break
            
    #with jsonlines.open(path_to_output, mode='w') as writer:
    #    writer.write_all(all_mapped_skills)
    
    return all_mapped_skills
 
def extract_from_jsonl(path_to_file,path_to_output,domain):
     
    all_mapped_skills = []
    counter = 0
    with jsonlines.open(path_to_file) as reader:
        for obj in reader:
            x =" ".join(word for word in obj['text'].split() if not word.startswith("\\"))
            text_input = TextInput(textEntered = x,domain=domain)
            print(text_input.textEntered)
            mapped_skills = get_mapped_skills2(text_input, method='llm')
            mapped_skills= mapped_skills.dict()
            mapped_skills['URL'] = x['URL']
            #mapped_skills['text'] = x
            all_mapped_skills.append(mapped_skills)
            # counter += 1    
            # print(counter)
            
            # if counter == 5:
            #     break
            
    with jsonlines.open(path_to_output, mode='w') as writer:
        writer.write_all(all_mapped_skills)
    
    return all_mapped_skills
        

def llm_wrapper(text,domain):
    text =" ".join(word for word in text.split() if not word.startswith("\\"))
    text_input = TextInput(textEntered = text,domain=domain)
    mapped_skills = get_mapped_skills2(text_input, method='llm')
    return mapped_skills

def extract_from_df(df,domain,path_to_output,n,start_index):
    
    list_df = [df[i:i+n] for i in range(0,len(df),n)]
    list_df_new = []
    for index, df_chunk in enumerate(list_df):
        if index>=start_index:
            file_name =f'jobs_extracted_skills_chunk_{index}.csv'
            output_path =path_to_output + file_name
            print(f"Processing chunk {index+1}")
            df_chunk['Skills'] = df_chunk.apply(lambda x: llm_wrapper(x['description'], domain),axis=1)
            df_chunk.to_csv(output_path,index=False)
            list_df_new.append(df_chunk)
    #df['Skills'] = df.apply(lambda x: llm_wrapper(x['description'], domain),axis=1)
    
    return list_df_new


if __name__ == '__main__':
    #text_files = '/home/gabor/Work/MAGAN/Abodoo/Data/Sample_texts/'
    #all_mapped_skills= batch_skill_extraction(text_files)
    #print(type(all_mapped_skills))
    #print(all_mapped_skills)
    
    base_dir ='/home/gabor/Work/MAGAN/Abodoo/Data/Greentech/Latest/'
    #file_name = 'filtered_urls_extracted_skills_FORMATED - filtered_urls_extracted_skills_FORMATED.csv'
    file_name = 'jobs_industries_salaries.csv'
    #path_to_file = '/home/gabor/Work/MAGAN/Abodoo/Data/Greentech/courses_corpus_data.jsonl'
    path_to_file = base_dir + file_name
    #path_to_output = '/home/gabor/Work/MAGAN/Abodoo/Data/Greentech/Latest/jobs_industries_salaries_extracted_skills.csv'
    path_to_output = '/home/gabor/Work/MAGAN/Abodoo/Data/Greentech/Latest/'
    domain = "greentech"
    chunk_size=50
    start_index = 8
    df = pd.read_csv(path_to_file)
    df_extended= extract_from_df(df,domain,path_to_output,chunk_size,start_index)
    
    #all_mapped_skills = extract_from_jsonl(path_to_file,path_to_output,domain)
    #all_mapped_skills = extract_from_csv(path_to_file,path_to_output,domain)
    