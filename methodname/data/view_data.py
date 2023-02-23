import joblib
import os
from tqdm import tqdm
import json
# import matplotlib.pyplot as plt

with open('temp' + '.pkl', 'rb') as f:
    data = joblib.load(f)
    print(data)

with open('temp.json','w') as f:
    json.dump(data,f)