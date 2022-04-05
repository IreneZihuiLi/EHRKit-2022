import sys
import os
import glob
import re
from getpass import getpass
try:
    from ehrkit import ehrkit
except ModuleNotFoundError:
    print('Looks like ehrkit was not installed yet. Run "python setup.py install" before running this script')
    exit()
from nltk.tokenize import sent_tokenize, word_tokenize

ehrdb = ehrkit.start_session(input("User?"), getpass("Password?"))
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




abb = input('What abbreviation are you looking for?')
outdirpath = cwd+('/output/reverse_abb/%s' %abb)
docids = ehrkit.get_documents_solr(abb)
try:
    os.makedirs(outdirpath)
    outpath = outdirpath
except FileExistsError:
    outpath = outdirpath

for i in docids:
    with open(os.path.join(outpath, '%d.txt' %i), 'w+') as f:
        text = ehrdb.get_document(i)
        f.write(text)
