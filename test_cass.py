
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "I am Chihaya Anon !!!!!"
encoded_input = tokenizer(text, return_tensors='pt')

print(encoded_input)


output = model(**encoded_input)

print(output[1].shape)
