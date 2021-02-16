import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import spacy
nlp =spacy.load('en_core_web_sm')


df = pd.read_csv("Uttarakhand.csv")
print(df.head())
questions = [
    'What is the Emergency helpline number?',
    'What happened in Uttarakhand?',
    'How many people are killed?',
    'Which tunnel completely blocked due to debris?'
]

df = df.drop_duplicates().reset_index(drop = True)
context = ''
for i in range(8):
    for text in df['tweet'][i]:
        context += text
print(context)

import re
def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)
final_context = deEmojify(context)
print(final_context)
context1 = ''
for i in range(75,85):
    for text in df['tweet'][i]:
        context1 += text
context1
print(context1)

from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


def answer_question(question, answer_text):
    input_ids = tokenizer.encode(question, answer_text)
    print('Query has {:,} tokens.\n'.format(len(input_ids)))
    sep_index = input_ids.index(tokenizer.sep_token_id)
    num_seg_a = sep_index + 1
    num_seg_b = len(input_ids) - num_seg_a
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    outputs = model(torch.tensor([input_ids]),
                    token_type_ids=torch.tensor([segment_ids]),
                    return_dict=True)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    answer = tokens[answer_start]

    for i in range(answer_start + 1, answer_end + 1):

        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]


        else:
            answer += ' ' + tokens[i]

    print('Answer: "' + answer + '"')

print(questions[0])
print(answer_question(questions[0], final_context))
print(questions[1])
print(answer_question(questions[1], final_context))
print(questions[2])
print(answer_question(questions[2], context1))
print(questions[3])
print(answer_question(questions[3], context1))