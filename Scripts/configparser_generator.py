from configparser import ConfigParser, ExtendedInterpolation
import os
import json
 
 
references = {}
references['soft_skills'] = ['ESCO','Lightcast']
references['hard_skills'] = ['ESCO','Lightcast'] 
references['domains'] = ['dashboard','analytics']

categories = {}
categories['dashboard'] = ['soft_skills','hard_skills','unmatched_skills']
categories['analytics'] = [
 'primary_hard_skill',
 'primary_qualification',
 'primary_soft_skill',
 'primary_transferable_skill',
 'related_hard_skill',
 'related_qualification',
 'related_soft_skill',
 'related_transferable_skill',
 ]

category_mapping = {}
category_mapping['primary_soft_skill'] = 'soft_skills'
category_mapping['primary_hard_skill'] = 'hard_skills'
category_mapping['related_soft_skill'] = 'soft_skills'
category_mapping['related_hard_skill'] = 'hard_skills'
category_mapping['primary_qualification'] = 'unmatched_skills'
category_mapping['primary_transferable_skill'] = 'unmatched_skills'
category_mapping['related_qualification'] = 'unmatched_skills'
category_mapping['related_transferable_skill'] = 'unmatched_skills'





# generic = {}
# generic['hard_skills_definition'] = '''Hard skills, or technical skills, are learned through education or hands-on experience.Examples of hard 
# skills include security complience, blueprint reading, programming, computer skills, data visualization or cost-benefit analysis. These are 
# concrete, measurable abilities that are often specific to a job.'''
# generic['soft_skills_definition'] = '''Soft skills include attributes and personality traits that help employees effectively interact with 
# others and succeed in the workplace. These skills can include social graces, communication abilities, personal habits, cognitive or emotional 
# empathy, time management, teamwork and leadesrship traits. The most important attributes are problem-solving, effective communication, self-direction,
# drive, adaptability, teamwork, dependability, conflict resolution,research, creativity, work ethic and integrity.'''

generic_prompt = '''Soft skills include attributes and personality traits that help employees effectively interact with 
others and succeed in the workplace. These skills can include social graces, communication abilities, personal habits, cognitive or emotional 
empathy, time management, teamwork and leadesrship traits. The most important attributes are problem-solving, effective communication, self-direction,
drive, adaptability, teamwork, dependability, conflict resolution,research, creativity, work ethic and integrity. Hard skills, or technical skills, 
are learned through education or hands-on experience.Examples of hard skills include security complience, blueprint reading, programming, computer 
skills, data visualization or cost-benefit analysis. These are concrete, measurable abilities that are often specific to a job or a domain.'''

# greentech = {}
# greentech['green_skills_definition'] = '''Green skills include technical knowledge, expertise and abilities that enable the effective use of 
# green technologies and processes in professional settings. They draw on a range of knowledge, values, and attitudes to facilitate environmentally 
# sustainable decision-making at work and in life. Examples of 'green knowledge' concepts include emission standards and ecological principles. 
# 'Green transversal skills' include, for instance, the evaluation of the environmental impact of personal behaviour and the adoption of ways 
# to boost biodiversity and animal welfare.'''

single_prompt = '''Soft skills include attributes and personality traits that help employees effectively interact with 
others and succeed in the workplace. These skills can include social graces, communication abilities, personal habits, cognitive or emotional 
empathy, time management, teamwork and leadesrship traits. The most important attributes are problem-solving, effective communication, self-direction,
drive, adaptability, teamwork, dependability, conflict resolution,research, creativity, work ethic and integrity. Hard skills, or technical skills, 
are learned through education or hands-on experience.Examples of hard skills include security complience, blueprint reading, programming, computer 
skills, data visualization or cost-benefit analysis. These are concrete, measurable abilities that are often specific to a job or a domain. 
Industry specific examples: Green skills include technical knowledge, expertise and abilities that enable the effective use of 
green technologies and processes in professional settings. They draw on a range of knowledge, values, and attitudes to facilitate environmentally 
sustainable decision-making at work and in life. Examples of 'green knowledge' concepts include emission standards and ecological principles. 
'Green transversal skills' include, for instance, the evaluation of the environmental impact of personal behaviour and the adoption of ways 
to boost biodiversity and animal welfare.  '''
 
def create_config():
    config = ConfigParser(interpolation=ExtendedInterpolation())
 
    # Add sections and key-value pairs
    config['general'] = {}
    config['general']['log_level'] = 'info'
    config['general']['domain_preference'] ='True'
    config['general']['domain'] = 'greentech'
    
    config['paths'] = {'base_dir': '.', 
                       'data_dir': '%(base_dir)s/Data/',
                       }
    
    
    
    config['dashboard'] = {}
    config['dashboard']['categories']=json.dumps(['hard_skills','soft_skills','unmatched_skills'])
    config['dashboard']['skill_def'] = '''Soft skills include attributes and personality traits that help employees effectively interact with 
    others and succeed in the workplace. These skills can include social graces, communication abilities, personal habits, cognitive or emotional 
    empathy, time management, teamwork and leadesrship traits. The most important attributes are problem-solving, effective communication, self-direction,
    drive, adaptability, teamwork, dependability, conflict resolution,research, creativity, work ethic and integrity. Hard skills, or technical skills, 
    are learned through education or hands-on experience.Examples of hard skills include security complience, blueprint reading, programming, computer 
    skills, data visualization or cost-benefit analysis. These are concrete, measurable abilities that are often specific to a job or a domain.'''
    
    config['greentech'] = {}
    config['greentech']['categories']=json.dumps(['primary_soft_skill','primary_hard_skill','related_soft_skill','related_hard_skill','primary_qualification','primary_transferable_skill','related_qualification','related_transferable_skill'])
    config['greentech']['skill_def'] = '''Green skills include technical knowledge, expertise and abilities that enable the effective use of green technologies and processes in professional settings. They draw on a range of knowledge, 
    values, and attitudes to facilitate environmentally sustainable decision-making at work and in life. Examples of 'green knowledge' concepts include emission standards and ecological principles.
    'Green transversal skills' include, for instance, the evaluation of the environmental impact of personal behaviour and the adoption of ways 
    to boost biodiversity and animal welfare. Green skills may be desribed by more than one or two words.'''
    
    config['mapper_config'] = {}
    config['mapper_config']['path'] = './Models/Neofuzz/'
    config['mapper_config']['mapping_similarity_threshold']=  '55'
    config['mapper_config']['maximum_score_diff']=  '5'
    config['mapper_config']['embedding_model'] = 'all-mpnet-base-v2'
    config['mapper_config']['references'] = json.dumps(references)
    config['mapper_config']['categories'] = json.dumps(categories)
    config['mapper_config']['category_mapping'] = json.dumps(category_mapping)
    config['mapper_config']['domain_preference'] = 'True'
    
    config['openai_config'] = {}
    config['openai_config']['endpoint'] = "os.getenv('AZURE_OPENAI_ENDPOINT')"
    config['openai_config']['api_key'] = "os.getenv('AZURE_OPENAI_API_KEY')"
    config['openai_config']['api_version'] = "2024-02-01"
    config['openai_config']['model'] = "gpt-4o"
    
    
    config['gliner_config'] = {}
    config['gliner_config']['path'] = './Models/GLiNER/large_v21_general/'
    config['gliner_config']['threshold'] = '0.1'
    
    config['llm_config'] = {}
    #config['llm_config']['generic'] = json.dumps(generic)
    #config['llm_config']['greentech'] = json.dumps(greentech)
    #config['llm_config']['hard_skills_definition'] =  "Hard skills, or technical skills, are learned through education or hands-on experience.Examples of hard skills include security complience, blueprint reading, programming, computer skills, data visualization or cost-benefit analysis. These are concrete, measurable abilities that are often specific to a job."
    #config['llm_config']['soft_skills_definition'] = '''Soft skills include attributes and personality traits that help employees effectively interact with others and succeed in the workplace. 
    #These skills can include social graces, communication abilities, personal habits, cognitive or emotional empathy, time management, teamwork and leadesrship traits.
    #The most important attributes are problem-solving, effective communication, self-direction,drive, adaptability, teamwork, dependability, conflict resolution,research, creativity, work ethic and integrity.'''
    config['llm_config']['single_prompt'] = single_prompt
    
    # Write the configuration to a file
    with open('./config.ini', 'w') as configfile:
        config.write(configfile)
 
 
if __name__ == "__main__":
    
    create_config()