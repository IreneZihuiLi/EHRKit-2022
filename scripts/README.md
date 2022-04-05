# Scripts usage
Before using any of the following, remember to activate your virtual environment
```
source /path/to/venv
```
**NOTE** Script 1,6 indexes data to Solr, while Script 5 creates a new table in the mimic database. Script 5 and 6 are already run and so should not be run again. Script 1 depending on future usage.

### 1. abb_extraction_solr.py
This script extracts abbreviations from texts of the MIMIC NOTEEVENTS table, and index the abbreviations to solr.
Currently this script indexes the extracted abbreviations(stored) and the text(not stored). If you want to retrieve the text, you can use the "getDocuments" function in ehrkit to retrieve text with the returned id.

You'll be prompted for username and password for mysql, as well as the number of documents you want to index

Usage:
```
python abb_extraction_solr.py
Number of Documents?[INPUT NUMDOCs here]
User?[INPUT MySQL DB username here]
Password?[INPUT MySQL DB password here]
```
**IMPORTANT:** After running the script, please go to the Solr Admin at http://tangra.cs.yale.edu:8983/solr/#/~cores/ehr_abbs_mimic and click "reload" in order to apply the new changes to the search index.

### 2. abb_extraction.py
This script extracts abbreviations from documents outputted to the directory EHRKit/output/discharge_summary and outputs the (abbreviations, sentence_id) pairs in **EHRKit/output/abbreviations/[DOCID].txt**

Usage
```
python abb_extraction.py
```

**NOTE:** You must output discharge summaries before running this script
```
ehrdb.output_note_events_discharge_summary()
```

### 3. find_abb_docs.py
This script finds documents that contains certain abbreviations, and outputs their text into **EHRKit/output/reverse_abb/[Abbreviation]/[DOCID].txt**

Usage:
```
python find_abb_docs.py
User?[INPUT MySQL DB username here]
Password?[INPUT MySQL DB password here]
What abbreviation are you looking for?[INPUT ABBREVIATION here]
```
### 4. train_word2vec.py
This script trains a word embedding model with gensim word2vec using data in **EHRKit/output/discharge_summary.**, and outputs the model to **EHRKit/models/discharge_model.**

Usage:

  1. Running this script
```
python train_word2vec.py
```
  2. Call this script in ehrkit by
  ```
  ehrkit.init_embedding_model()
  ```


### 5. tables.py
This script initializes the SENT_SEGS table in the MIMIC database.
Table Schema: ROW_ID, DOC_ID, SENT_ID, START_BYTE, NUM_BYTES
### 6. create_solr_data_umn.py
This script creates json files for UMN Clinical Abbreviation Sense Inventory to the directory "/data/projects/EHR-NLP/data/solr_data/".

Current usage:
FILEPATH is the path to the dataset file
```
python create_solr_data_umn FILEPATH
```
To index those files to solr, we can use the post tool
```
cd /path/to/solr_data/parent/directory
ln -s /data/tools/solr-7.4.0/bin/post post
for i in `ls solr_data/*.json`; do ./post -c ehr_abbsense_umn $i;done
```
**TODO**: Might want to skip creating files and directly post dictionary to solr core
