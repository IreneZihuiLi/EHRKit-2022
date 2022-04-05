import sys
import os
import glob
import re
try:
    from ehrkit import ehrkit
except ModuleNotFoundError:
    print('Looks like ehrkit was not installed yet. Run "python setup.py install" before running this script')
    exit()
from nltk.tokenize import sent_tokenize, word_tokenize
#ehrdb = ehrkit.start_session(input('user'), input('password'))

def output_abb():
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(cwd,'output/discharge_summary/*.txt')
    outdirpath = cwd+('/output/abbreviations')
    try:
        os.makedirs(outdirpath)
        outpath = outdirpath
    except FileExistsError:
        outpath = outdirpath

    files = glob.glob(path)
    raw_docs = []

    for name in files:
        try:
            with open(name) as f:
                raw_t = f.read()
                abbreviations_raw = ehrkit.get_abbs_sent_ids(raw_t)
                abbreviations = [abb[0]+','+str(abb[1])+'\n' for abb in abbreviations_raw]
            with open(os.path.join(outpath, os.path.basename(name)), 'w+') as f:
                f.writelines(abbreviations)
                print("output abbreviations in {}".format(os.path.basename(name)))

        except KeyboardInterrupt:
            exit()
        except FileNotFoundError:
            print("File not found!")

if __name__ == '__main__':
    output_abb()
