import string

# simple character tokenizer
class SimpleTokenizer:
    def __init__(self):
        self.vocab = [
            '-',
            '|',
        ]

        # add lower case letters
        self.vocab += string.ascii_lowercase

        # create id to symbol mapping
        self.id2sym = {i: s for i, s in enumerate(self.vocab)}

        # create symbol to id mapping
        self.sym2id = {s: i for i, s in enumerate(self.vocab)}
    
    def encode_as_ids(self, text):
        """returns list of integer ids for each character in text"""
        return [self.sym2id[s] for s in text]

    def decode_ids(self, ids):
        """returns text from list of integer ids"""
        return ''.join([self.id2sym[i] for i in ids])