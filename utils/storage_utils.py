"""
This file contains utilities for FS handling

"""
import os
import json

def createdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def writejson(filename,data):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)