
from pytorch_pretrained_bert.tokenization import BertTokenizer



def whitespace_tokenize(text):
    """
    Desc:
        runs basic whitespace cleaning and splitting on a piece of text
    """
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens



class BertTokenizer4Tagger(BertTokenizer):
    """
    Desc:
        slove the problem of tagging span can not fit after run word_piece tokenizing
    """
    def __init__(self, vocab_file, do_lower_case=False, max_len=None,
        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):

        super(BertTokenizer4Tagger, self).__init__(vocab_file, do_lower_case=do_lower_case,
            max_len=max_len, never_split=never_split)


    def tokenize(self, text, label_lst=None):
        """
        Desc:
            text:
            label_lst: ["B", "M", "E", "S", "O"]
        """

        split_tokens = []
        split_labels = []

        if label_lst is None:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
            return split_tokens


        for token, label in zip(self.basic_tokenizer.tokenize(text), label_lst):
            # cureent token should be 1 single word
            sub_tokens = self.wordpiece_tokenizer.tokenize(token)
            if len(sub_tokens) > 1:
                for tmp_idx, tmp_sub_token in enumerate(sub_tokens):
                    if tmp_idx == 0:
                        split_tokens.append(tmp_sub_token)
                        split_labels.append(label)
                    else:
                        split_tokens.append(tmp_sub_token)
                        split_labels.append("X")
            else:
                split_tokens.append(sub_token)
                split_labels.append(label)

        return split_tokens, split_labels


def load_data(sent, bert_path):
    tokenizer=BertTokenizer4Tagger.from_pretrained(bert_path, do_lower_case=True)

    query_tokens = tokenizer.tokenize("项目名")
    whitespace_doc = whitespace_tokenize(sent)
    all_doc_tokens = []

    for token_item in whitespace_doc:
        tmp_subword_lst = tokenizer.tokenize(token_item)
        all_doc_tokens.extend(tmp_subword_lst)

    input_tokens = ["[CLS]"]
    segment_ids = [0]

    for query_item in query_tokens:
        input_tokens.append(query_item)
        segment_ids.append(0)

    input_tokens.append("[SEP]")
    segment_ids.append(0)

    input_tokens.extend(all_doc_tokens)
    segment_ids.extend([1]* len(all_doc_tokens))
    input_tokens.append("[SEP]")
    segment_ids.append(1)
    input_mask = [1] * len(input_tokens)




    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    max_seq_length=100
    if len(input_ids) < max_seq_length:
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

    return input_tokens, input_ids, input_mask, segment_ids

