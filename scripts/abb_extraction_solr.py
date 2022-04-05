import sys
import os
import json
from getpass import getpass
try:
    from ehrkit import ehrkit
except ModuleNotFoundError:
    print('Looks like ehrkit was not installed yet. Run "python setup.py install" before running this script')
    exit()
#ehrkit.post_single_dict_to_solr(d, core)
def index_abb(numDOCs):
    ehrdb = ehrkit.start_session(input('user'), getpass('password'))
    for i in range(1,numDOCs+1):
        try:
            text = ehrdb.get_document(i)
            abbreviations_raw = ehrkit.get_abbs_sent_ids(text)
            d = { 'id': i }
            d['abbreviations'] = [abb[0] for abb in abbreviations_raw]
            d['doctext_not_stored'] = text
            #d['doctext'] = text
           #d['abbreviations_sent_ids'] = [abb[0]+':'+str(abb[1]) for abb in abbreviations_raw]
            #json_d = json.dump([{"abbreviations":abb[0]} for abb in abbreviations_raw])
            ehrkit.post_single_dict_to_solr(d, 'ehr_abbs_mimic')
            print('Indexed abbreviations in document %d' %i)
        except KeyboardInterrupt:
            exit()


if __name__ == '__main__':
    index_abb(int(input('Number of Documents?')))
