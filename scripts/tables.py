import sys, os

sys.path.append(os.path.dirname(sys.path[0]))

from ehrkit import ehrkit
from config import USERNAME, PASSWORD

# Number of documents in NOTEEVENTS.
NUM_DOCS = 2083180

def create_table(db, name, primary_keys, columns):

    ''' Format of columns: [(column_name, data_type), ...]
        Format of primary_keys: tuple of key(s)

        Returns 1 if table created successfully (data to be inserted).
        Returns 0 if table not created successfully (data will NOT be inserted).
    '''

    db.cur.execute("SHOW TABLES")
    raw = db.cur.fetchall()

    tables = ehrkit.flatten(raw)

    if name in tables:
        print("Error: table {0} already exists. Not inserting any data.".format(name))

        return 0
    else:
        try:
            db.cur.execute("CREATE TABLE {0} (ROW_ID INT AUTO_INCREMENT, UNIQUE KEY (row_id))".format(name))
            
            for column in columns:
                db.cur.execute("ALTER TABLE {0} ADD COLUMN {1} {2}".format(name, column[0], column[1]))

            keys = ", ".join(primary_keys)
            keys = "({0})".format(keys)
            db.cur.execute("ALTER TABLE {0} ADD PRIMARY KEY {1}".format(name, keys))

            return 1

        except Exception as e:
            print(e)
            drop_table(db, name)

            return 0

def drop_table(db, name):
    print("Deleting table {0}".format(name))
    db.cur.execute("DROP TABLE {0}".format(name))

def fill_table(db, name, columns, data):

    ''' Format of columns: [(column_name, data_type), ...]
        Format of data: [(column1, column2, ...), (column1, column2, ...), ...]
    '''

    # Format cols to be used in SQL command
    col_names = [column[0] for column in columns]
    cols = ", ".join(col_names)
    cols = "({0})".format(cols)

    # Make list of string formatting shortcuts to be used in SQL command
    format_list = []
    for column in columns:
            format_list.append(r"%s")
    formats = ", ".join(format_list)
    formats = "({0})".format(formats)

    print("Beginning to fill table {0}. This may take a while".format(name))

    sql = "INSERT INTO {0} {1} VALUES {2}".format(name, cols, formats)
    db.cur.executemany(sql, data)
    
    db.cnx.commit()

def index_doc(db, doc_id):
    ''' Returns [(doc_id, sent_id, start_byte, num_bytes), ...] for filling SENT_SEGS.
    '''
    output = []

    text = db.get_document(doc_id)
    sents = db.get_document_sents(doc_id)

    start = 0
    numbytes = 0

    for i, sent in zip(range(0, len(sents)), sents):
        start += text[start:].find(sent)
        numbytes = len(sent)
        output.append((doc_id, i, start, numbytes))
    
    return output

def retrieve_abbrevs(db, doc_id):
    ''' Returns [(abbreviation, doc_id, sent_id), ...] for filling ABBREVIATION_INDICES.
    '''
    output = []

    abbrevs = db.get_abbreviation_sent_ids(doc_id)

    for abbrev in abbrevs:
        output.append((abbrev[0], doc_id, abbrev[1]))
    return output

def retrieve_pat_ids(db, doc_id):
    ''' Returns [(doc_id, patient_id), ...] for filling PATIENT_INDICES.
    '''

    db.cur.execute("SELECT SUBJECT_ID FROM mimic.NOTEEVENTS WHERE ROW_ID = {0}".format(doc_id))
    raw = db.cur.fetchall()
    pat_id = raw[0][0]

    output = [(doc_id, pat_id),]

    return output

def upload_table(db, name, keys, columns, row_num, func):
    ''' Calls create_table() and fill_table(). Only needs to be called once per table.
        Returns 1 if successful, 0 if not.

        Parameter func is the method that retrieves the data for the table being created.
    '''
    print("Organizing data for {0}. This may take a while".format(name))

    data = []

    if create_table(db, name, keys, columns):
        for i in range(1, row_num + 1):
            print("Processing row {0}...".format(i))
            data.extend(func(db, i))
        fill_table(db, name, columns, data)
        print("Table {0} successfully created and filled.".format(name))
        return 1
    else:
        print("Table {0} not created.".format(name))
        return 0


if __name__ == '__main__':
    ehrdb = ehrkit.start_session(USERNAME, PASSWORD)

    char_type = "varchar(255)"
    int_type = "int"

    data = []

    ### Creating and filling SENT_SEGS. ###
    sent_segs_keys = ("DOC_ID", "SENT_ID")
    sent_segs_columns = [("DOC_ID", int_type), ("SENT_ID", int_type), ("START_BYTE", int_type), ("NUM_BYTES", int_type)]

    upload_table(ehrdb, "SENT_SEGS", sent_segs_keys, sent_segs_columns, NUM_DOCS, index_doc)

    ### Creating and filling ABBREVIATION_INDICES ###
    abb_keys = ("ROW_ID", "ABBREVIATION")
    abb_columns = [("ABBREVIATION", char_type), ("DOC_ID", int_type), ("SENT_ID", int_type)]

    upload_table(ehrdb, "ABBREVIATION_INDICES", abb_keys, abb_columns, NUM_DOCS, retrieve_abbrevs)
    
    ### Creating and filling PATIENT_INDICES. ###

    pat_keys = ("DOC_ID", "PATIENT_ID")
    pat_columns = [("DOC_ID", int_type), ("PATIENT_ID", int_type)]

    upload_table(ehrdb, "PATIENT_INDICES", pat_keys, pat_columns, NUM_DOCS, retrieve_pat_ids)