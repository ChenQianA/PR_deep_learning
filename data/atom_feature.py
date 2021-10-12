import urllib
import json
import numpy as np

ATOM_INIT_JSON_URL = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/atom_init.json'
urllib.request.urlretrieve(ATOM_INIT_JSON_URL, "atom_feature.json")

with open("atom_feature.json","rb") as f:
    atom_feature=json.load(f)
    
af = np.array(list(atom_feature.values()))

with open("atom_init.json","rb") as f:
    atom_init=json.load(f)
    
ai = np.array(list(atom_init.values()))

if np.equal(af,ai).all():
    af = np.concatenate([np.zeros((1, af.shape[1])), af], axis=0, dtype=np.float32) 
    
np.save('atom_init.npy', af)
