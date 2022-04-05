from transformers import MarianMTModel, MarianTokenizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from utils import get_sents_stanza, get_multiple_sents_stanza
from transformers import pipeline

LANG_CODE = {'Malay_written_with_Latin': '>>zlm_Latn<<', 'Mauritian_Creole': '>>mfe<<', 'Haitian': '>>hat<<',
             'Papiamento': '>>pap<<', 'Asturian': '>>ast<<', 'Catalan': '>>cat<<', 'Indonesian': '>>ind<<',
             'Galician': '>>glg<<', 'Walloon': '>>wln<<', 'Spanish': '>>spa<<', 'French': '>>fra<<',
             'Romanian': '>>ron<<', 'Portuguese': '>>por<<', 'Italian': '>>ita<<', 'Occitan': '>>oci<<',
             'Aragonese': '>>arg<<', 'Minangkabau': '>>min<<'}

def get_supported_translation_languages():
    return list(LANG_CODE.keys())

def get_translation(text, model_name, target_language):
    '''
    returns a string, which is the translated version of text
    '''

    # logging
    print(f'Translating medical note using {model_name}')
    partial_input = '\n'.join(text.split('\n')[:5])
    print(f"Input text (truncated): {partial_input}\n...")

    # translation using MarianMT
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    sents = [f'{LANG_CODE[target_language]} ' + sent for sent in get_sents_stanza(text)]

    translated = model.generate(**tokenizer(sents, return_tensors="pt", padding=True))
    translated = ' '.join([tokenizer.decode(t, skip_special_tokens=True) for t in translated])

    return translated

def get_bert_embeddings(pretrained_model, texts):
    """
    texts: a list of lists of sentences, each list is made up of sentences from the same document
    """

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = AutoModel.from_pretrained(pretrained_model, return_dict=False, output_hidden_states=True)

    output_embeddings = []

    for text in texts:
        # get embeddings for each sentence, compute mean to represent document
        inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')

        with torch.no_grad():
            model_output = model(**inputs)

        pooled_output = model_output[1]
        mean_embedding = torch.mean(pooled_output, axis=0)

        output_embeddings.append(mean_embedding.numpy())

    return output_embeddings

def get_single_summary(text, model_name="t5-small", min_length=50, max_length=200):
    '''
    https://huggingface.co/transformers/v3.0.2/_modules/transformers/pipelines.html#SummarizationPipeline
    :param text: input sequence, a string or a list of string
    :param model_name: model_name: `bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`
    :param min_length: min length in summary
    :param max_length: max length in summary
    :return: summary string
    '''
    # choices: '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'

    classifier = pipeline("summarization", model=model_name, tokenizer=model_name)
    res = classifier(text, min_length=min_length, max_length=max_length)
    final_summary = []

    for summary in res:
        final_summary.append(summary['summary_text'])

    final_summary = '\n\n'.join(final_summary)

    return final_summary

def get_multi_summary_joint(text, model_name="osama7/t5-summarization-multinews", min_length=50, max_length=200):
    '''
    Join all the input documents as a long document, then do single document summarization
    :param text: input sequence, a string or a list of string
    :param model_name: model_name: `bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`, other models https://huggingface.co/models?sort=downloads&search=summarization
    :param min_length: min length in summary
    :param max_length: max length in summary
    :return: summary string
    '''
    # choices: '`bart-large-cnn`', '`t5-small`', '`t5-base`', '`t5-large`', '`t5-3b`', '`t5-11b`'

    classifier = pipeline("summarization", model=model_name, tokenizer=model_name)
    text = ' '.join(text)
    res = classifier(text, min_length=min_length, max_length=max_length)
    final_summary = []

    for summary in res:
        final_summary.append(summary['summary_text'])

    final_summary = '\n\n'.join(final_summary)

    return final_summary
