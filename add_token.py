from transformers import PreTrainedTokenizerFast
from transformers import GPT2LMHeadModel

with open("C:/kogpt2_ft/new_list.txt", "r", encoding = "utf-8") as f:
    lines = f.readlines()

a = [line.strip() for line in lines]
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token = '</s>', eos_token = '</s>', unk_token = '<unk>', pad_token = '<pad>', mask_token = '<mask>')

c_tokens = a
tokenizer.add_tokens(c_tokens)
tokenizer.save_pretrained("add_new_token")
new_tokenizer = PreTrainedTokenizerFast.from_pretrained("add_new_token")
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
model.resize_token_embeddings(len(new_tokenizer))
print(new_tokenizer.tokenize("DB하이텍의 유동비율은 얼마야?"))