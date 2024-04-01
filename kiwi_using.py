from kiwipiepy import Kiwi
kiwi = Kiwi()
result = kiwi.tokenize("공부하기 싫다")
print(result)
import pandas as pd
train = pd.read_csv('C:/kogpt2_ft/chat_train.csv')
train
r = kiwi.tokenize("DB하이텍의 2011년 12월 유동비율은 얼마야")
print(r)
vocab = pd.read_csv('C:/kogpt2_ft/user_dic.tsv', sep = '\t', header = None)
vocab
vocab.to_csv('C:/kogpt2_ft/vocab.csv')
vocab
import sentencepiece
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
to = tokenizer.tokenize("DB하이텍의 유동비율 알려줘")
print(to)