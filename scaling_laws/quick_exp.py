import json
import copy
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()
                                                                
from cs336_scaling.client import submit_experiment, get_budget              
cfg = json.load(open('scaling_laws/configs/lr_88m_5760_lr0p01.json'))       
print('budget before:', get_budget())                                       
print('submitted:', submit_experiment(cfg))                                 
print('budget after:', get_budget())                                        