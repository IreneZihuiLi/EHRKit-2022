from scispacy_functions import get_abbreviations, get_hyponyms, get_linked_entities, get_named_entities
from transformer_functions import get_translation, get_supported_translation_languages, get_single_summary, get_multi_summary_joint
from utils import get_sents_stanza, get_multiple_sents_stanza, get_sents_pyrush, get_sents_scispacy
from multi_doc_functions import get_clusters, get_similar_documents
import numpy as np
from stanza_functions import (
    get_named_entities_stanza_biomed,
    get_sents_stanza_biomed,
    get_tokens_stanza_biomed,
    get_part_of_speech_and_morphological_features,
    get_lemmas_stanza_biomed,
    get_dependency_stanza_biomed
)

class EHRKit:
    """
    EHRKit is the main class of this toolkit. An EHRKit object stores textual records and default models for various tasks.
    Different tasks can be called from an EHRKit object to perform tasks on the stored textual records.

    Args:
        main_record (str): main textual record
        support_records (list): list of auxiliary textual records (used in multi-document tasks)
        scispacy_model (str): default model for scispacy tasks
        bert_model (str): default model for pre-trained transformers
        marian_model (str): default model for translation
    """
    def __init__(self,
                 main_record="",
                 supporting_records=[],
                 scispacy_model="en_core_sci_sm",
                 bert_model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                 marian_model="Helsinki-NLP/opus-mt-en-ROMANCE"):

        self.main_record = main_record
        self.supporting_records = supporting_records
        self.scispacy_model = scispacy_model
        self.bert_model = bert_model
        self.marian_model = marian_model

    ''' Functions for manipulating records and default models '''
    def update_and_delete_main_record(self, main_record):
        """
        update current main_record, current main_record will be deleted
        """
        if main_record:
            self.main_record = main_record
        else:
            raise TypeError('Invalid type for main_record')

    def update_and_keep_main_record(self, main_record):
        """
        update current main_record, current main_record will be added to the end of supporting_records
        """
        if main_record:
            self.supporting_records.append(self.main_record)
            self.main_record = main_record
        else:
            raise TypeError('Invalid type for main_record')

    def replace_supporting_records(self, supporting_records):
        """
        replace current supporting records
        """
        self.supporting_records = supporting_records

    def add_supporting_records(self, supporting_records):
        """
        add additional supporting records to existing supporting records
        """
        self.supporting_records.extend(supporting_records)

    def update_scispacy_model(self, scispacy_model):
        self.scispacy_model = scispacy_model

    def update_bert_model(self, bert_model):
        self.bert_model = bert_model

    def update_marian_model(self, marian_model):
        self.marian_model = marian_model

    ''' Functions for textual record processing '''
    def get_abbreviations(self):
        abbreviations = get_abbreviations(self.scispacy_model, self.main_record)
        return abbreviations

    def get_hyponyms(self):
        hyponyms = get_hyponyms(self.scispacy_model, self.main_record)
        return hyponyms

    def get_linked_entities(self):
        linked_entities = get_linked_entities(self.scispacy_model, self.main_record)
        return linked_entities

    def get_named_entities(self, tool='scispacy'):
        if tool ==  'scispacy':
            named_entities = get_named_entities(self.scispacy_model, self.main_record)
        elif tool == 'stanza':
            named_entities = get_named_entities_stanza_biomed(self.main_record)
        return named_entities

    def get_translation(self, target_language='Spanish'):
        translation = get_translation(self.main_record, self.marian_model, target_language)
        return translation

    def get_supported_translation_languages(self):
        return get_supported_translation_languages()

    def get_sentences(self, tool='stanza'):
        if tool == 'pyrush':
            sents = get_sents_pyrush(self.main_record)
            sents = [self.main_record[sent.begin:sent.end] for sent in sents]
        elif tool == 'stanza':
            sents = get_sents_stanza(self.main_record)
        elif tool == 'scispacy':
            sents = get_sents_scispacy(self.main_record)
        elif tool == 'stanza-biomed':
            sents = get_sents_stanza_biomed(self.main_record)
        return sents

    def get_tokens(self, tool='stanza-biomed'):
        if tool == 'stanza-biomed':
            tokens = get_tokens_stanza_biomed(self.main_record)
        return tokens

    def get_pos_tags(self, tool='stanza-biomed'):
        if tool ==  'stanza-biomed':
            tags = get_part_of_speech_and_morphological_features(self.main_record)
        return tags

    def get_lemmas(self, tool='stanza-biomed'):
        if tool == 'stanza-biomed':
            lemmas = get_lemmas_stanza_biomed(self.main_record)
        return lemmas

    def get_dependency(self, tool='stanza-biomed'):
        if tool == 'stanza-biomed':
            dependencies = get_dependency_stanza_biomed(self.main_record)
        return dependencies

    def get_clusters(self, k=2):
        # combine main record and candidate records for clustering
        docs = [self.main_record] + self.supporting_records
        clusters = get_clusters(self.bert_model, docs, k)
        return clusters

    def get_similar_documents(self, k=2):
        query_note = self.main_record
        candidate_notes = self.supporting_records
        # ids of candidates
        candidates = np.array(range(len(candidate_notes)))
        similar_docs = get_similar_documents(self.bert_model, query_note, candidate_notes, candidates, top_k=k)
        return similar_docs

    def get_single_record_summary(self):
        summary = get_single_summary(self.main_record)
        return summary

    def get_multi_record_summary(self):
        docs = [self.main_record] + self.supporting_records
        summary = get_multi_summary_joint(docs)
        return summary
