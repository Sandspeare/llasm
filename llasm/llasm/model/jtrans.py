import torch
from collections import defaultdict
from transformers import BertTokenizer, BertForMaskedLM, BertModel

vocab_data = open("../../encoder/jtrans/tokenizer/vocab.txt").read().strip().split("\n") + ["[SEP]", "[PAD]", "[CLS]", "[MASK]"]
my_vocab = defaultdict(lambda: 512, {vocab_data[i] : i for i in range(len(vocab_data))})

def help_tokenize(split_line):
    global my_vocab
    ret = {}
    split_line_len = len(split_line)
    if split_line_len <= 509:
        split_line = ['[CLS]']+split_line+['[SEP]']
        attention_mask = [1] * len(split_line) + [0] * (512 - len(split_line))
        split_line = split_line + (512-len(split_line))*['[PAD]']
    else:
        split_line = ['[CLS]'] + split_line[:510] + ['[SEP]']
        attention_mask = [1]*512
    input_ids = [my_vocab[e] for e in split_line]
    ret['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
    ret['attention_mask'] = torch.tensor(attention_mask, dtype=torch.long)
    return ret

class BinBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.embeddings.position_embeddings=self.embeddings.word_embeddings