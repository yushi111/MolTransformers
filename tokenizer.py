"""SMILES Tokenizer module.
"""
import re
import json
import warnings
from re import Pattern, template
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union
from typing import Any

import torch
from torch.utils.data import TensorDataset,DataLoader,random_split

Tokens = List[str]


class SMILESTokenizer:
    def __init__(
        self,
        smiles: List[str] = None,
        tokens: List[str] = None,
        regex_token_patterns: List[str] = None,
        beginning_of_smiles_token: str = "^",
        end_of_smiles_token: str = "&",
        padding_token: str = " ",
        unknown_token: str = "?",
        encoding_type: str = "index",  # "one hot" or "index"
        filename: str = None,
    ) -> None:
        self._check_encoding_type(encoding_type)

        self.encoding_type = encoding_type
        self._beginning_of_smiles_token = beginning_of_smiles_token
        self._end_of_smiles_token = end_of_smiles_token
        self._padding_token = padding_token
        self._unknown_token = unknown_token

        self._regex_tokens: List[str] = []
        self._tokens: List[str] = []

        smiles = smiles or []
        regex_token_patterns = regex_token_patterns or []
        tokens = tokens or []

        with warnings.catch_warnings(record=smiles != [] or filename):
            self.add_regex_token_patterns(regex_token_patterns)
            self.add_tokens(tokens)

        self._re: Optional[Pattern] = None
        self._vocabulary: Dict[str, int] = {}
        self._decoder_vocabulary: Dict[int, str] = {}
        if smiles:
            self.create_vocabulary_from_smiles(smiles)
        elif filename:
            self.load_vocabulary(filename)

    @property
    def special_tokens(self) -> Dict[str, str]:
        """Returns a dictionary of non-character tokens"""
        return {
            "start": self._beginning_of_smiles_token,
            "end": self._end_of_smiles_token,
            "pad": self._padding_token,
            "unknown": self._unknown_token,
        }

    @property
    def vocabulary(self) -> Dict[str, int]:
        """Tokens vocabulary.
        :return: Tokens vocabulary
        """
        if not self._vocabulary:
            self._vocabulary = self._reset_vocabulary()
        return self._vocabulary

    @property
    def decoder_vocabulary(self) -> Dict[int, str]:
        """Decoder tokens vocabulary.
        :return: Decoder tokens vocabulary
        """
        if not self._decoder_vocabulary:
            self._decoder_vocabulary = self._reset_decoder_vocabulary()
        return self._decoder_vocabulary

    @property
    def re(self) -> Pattern:
        """Tokens Regex Object.
        :return: Tokens Regex Object
        """
        if not self._re:
            self._re = self._get_compiled_regex(self._tokens, self._regex_tokens)
        return self._re

    def __call__(
        self, data: Union[str, List[str]], *args, **kwargs
    ) -> List[torch.Tensor]:
        return self.encode(data, *args, **kwargs)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __getitem__(self, item: str) -> int:
        if item in self.special_tokens:
            return self.vocabulary[self.special_tokens[item]]
        if item not in self.vocabulary:
            raise KeyError(f"Unknown token: {item}")
        return self.vocabulary[item]

    def _reset_vocabulary(self) -> Dict[str, int]:
        """Create a new tokens vocabulary.
        :return: New tokens vocabulary
        """
        return {
            self._padding_token: 0,
            self._beginning_of_smiles_token: 1,
            self._end_of_smiles_token: 2,
            self._unknown_token: 3,
        }

    def _reset_decoder_vocabulary(self) -> Dict[int, str]:
        """Create a new decoder tokens vocabulary.
        :return: New decoder tokens vocabulary
        """
        return {i: t for t, i in self.vocabulary.items()}

    def encode(
        self,
        data: Union[List[str], str],
        encoding_type: Optional[str] = None,
    ) -> List[torch.Tensor]:
        if encoding_type is None:
            encoding_type = self.encoding_type

        self._check_encoding_type(encoding_type)
        if isinstance(data, str):
            # Convert string to a list with one string
            data = [data]

        tokenized_data = self.tokenize(data)
        id_data = self.convert_tokens_to_ids(tokenized_data)
        encoded_data = self.convert_ids_to_encoding(id_data, encoding_type)

        return encoded_data

    def tokenize(self, data: List[str]) -> List[List[str]]:
        tokenized_data = []

        for smi in data:
            tokens = self.re.findall(smi)
            tokenized_data.append(
                [self._beginning_of_smiles_token] + tokens + [self._end_of_smiles_token]
            )

        return tokenized_data

    def convert_tokens_to_ids(self, token_data: List[List[str]]) -> List[torch.Tensor]:
        tokens_lengths = list(map(len, token_data))
        ids_list = []

        for tokens, length in zip(token_data, tokens_lengths):
            ids_tensor = torch.zeros(length, dtype=torch.long)
            for tdx, token in enumerate(tokens):
                ids_tensor[tdx] = self.vocabulary.get(
                    token, self.vocabulary[self._unknown_token]
                )
            ids_list.append(ids_tensor)

        return ids_list

    def convert_ids_to_encoding(
        self, id_data: List[torch.Tensor], encoding_type: Optional[str] = None
    ) -> List[torch.Tensor]:
        if encoding_type is None:
            encoding_type = self.encoding_type

        self._check_encoding_type(encoding_type)

        if encoding_type == "index":
            return id_data
        # Implies "one hot" encoding
        num_tokens = len(self.vocabulary)
        onehot_tensor = torch.eye(num_tokens)
        onehot_data = [onehot_tensor[ids] for ids in id_data]
        return onehot_data

    def decode(
        self, encoded_data: List[torch.Tensor], encoding_type: Optional[str] = None
    ) -> List[str]:
        id_data = self.convert_encoding_to_ids(encoded_data, encoding_type)
        tokenized_data = self.convert_ids_to_tokens(id_data)
        smiles = self.detokenize(tokenized_data)

        return smiles

    def detokenize(
        self,
        token_data: List[List[str]],
        include_control_tokens: bool = False,
        include_end_of_line_token: bool = False,
        truncate_at_end_token: bool = False,
    ) -> List[str]:
        character_lists = [tokens.copy() for tokens in token_data]

        character_lists = [
            self._strip_list(
                tokens,
                strip_control_tokens=not include_control_tokens,
                truncate_at_end_token=truncate_at_end_token,
            )
            for tokens in character_lists
        ]

        if include_end_of_line_token:
            for s in character_lists:
                s.append("\n")

        strings = ["".join(s) for s in character_lists]

        return strings

    def convert_ids_to_tokens(self, ids_list: List[torch.Tensor]) -> List[List[str]]:
        tokens_data = []
        for ids in ids_list:
            tokens = [self.decoder_vocabulary[i] for i in ids.tolist()]
            tokens_data.append(tokens)

        return tokens_data

    def convert_encoding_to_ids(
        self, encoded_data: List[torch.Tensor], encoding_type: Optional[str] = None
    ) -> List[torch.Tensor]:
        if encoding_type is None:
            encoding_type = self.encoding_type

        self._check_encoding_type(encoding_type)

        if encoding_type == "index":
            return encoded_data

        # Implies "one hot" encoding
        id_data = []
        for encoding in encoded_data:
            indices, t_ids = torch.nonzero(encoding, as_tuple=True)
            ids = torch.zeros(encoding.shape[0], dtype=torch.long)
            ids[indices] = t_ids
            id_data.append(ids)

        return id_data

    def add_tokens(self, tokens: List[str], regex: bool = False, smiles=None) -> None:
        existing_tokens = self._regex_tokens if regex else self._tokens
        for token in tokens:
            if token in existing_tokens:
                raise ValueError(f'"{token}" already present in list of tokens.')

        if regex:
            self._regex_tokens[0:0] = tokens
        else:
            self._tokens[0:0] = tokens

        # Get a compiled tokens regex
        self._re = self._get_compiled_regex(self._tokens, self._regex_tokens)

        if not smiles:
            warnings.warn(
                "Tokenizer vocabulary has not been updated. Call `create_vocabulary_from_smiles`\
                with SMILES data to update."
            )
        else:
            self.create_vocabulary_from_smiles(smiles)

    def add_regex_token_patterns(
        self, tokens: List[str], smiles: List[str] = None
    ) -> None:
        self.add_tokens(tokens, regex=True, smiles=smiles)

    def create_vocabulary_from_smiles(self, smiles: List[str]) -> None:
        # Reset Tokens Vocabulary
        self._vocabulary = self._reset_vocabulary()

        for tokens in self.tokenize(smiles):
            for token in tokens:
                self._vocabulary.setdefault(token, len(self._vocabulary))

        # Reset decoder tokens vocabulary
        self._decoder_vocabulary = self._reset_decoder_vocabulary()

    def remove_token_from_vocabulary(self, token: str) -> None:
        
        vocabulary_tokens: List[str] = list(self.vocabulary.keys())

        if token not in vocabulary_tokens:
            raise ValueError(f"{token} is not in the vocabulary")

        vocabulary_tokens.remove(token)

        # Recreate tokens vocabulary
        self._vocabulary = {t: i for i, t in enumerate(vocabulary_tokens)}

    def load_vocabulary(self, filename: str) -> None:
        """
        Load a serialized vocabulary from a JSON format
        :param filename: the path to the file on disc
        """
        with open(filename, "r") as fileobj:
            dict_ = json.load(fileobj)

        self._update_state(dict_["properties"])
        self._vocabulary = {token: idx for idx, token in enumerate(dict_["vocabulary"])}
        self._reset_decoder_vocabulary()

    def save_vocabulary(self, filename: str) -> None:
        
        token_tuples = sorted(self.vocabulary.items(), key=lambda k_v: k_v[1])
        tokens = [key for key, _ in token_tuples]
        dict_ = {"properties": self._state_properties(), "vocabulary": tokens}
        with open(filename, "w") as fileobj:
            json.dump(dict_, fileobj, indent=4)

    def _strip_list(
        self,
        tokens: List[str],
        strip_control_tokens: bool = False,
        truncate_at_end_token: bool = False,
    ) -> List[str]:
        """Cleanup tokens list from control tokens.
        :param tokens: List of tokens
        :param strip_control_tokens: Flag to remove control tokens, defaults to False
        :param truncate_at_end_token: If True truncate tokens after end-token
        """
        if truncate_at_end_token and self._end_of_smiles_token in tokens:
            end_token_idx = tokens.index(self._end_of_smiles_token)
            tokens = tokens[: end_token_idx + 1]

        strip_characters: List[str] = [self._padding_token]
        if strip_control_tokens:
            strip_characters.extend(
                [self._beginning_of_smiles_token, self._end_of_smiles_token]
            )
        while tokens[0] in strip_characters:
            tokens.pop(0)

        reversed_tokens: Iterator[str] = reversed(tokens)

        while next(reversed_tokens) in strip_characters:
            tokens.pop()

        return tokens

    def _get_compiled_regex(
        self, tokens: List[str], regex_tokens: List[str]
    ) -> Pattern:
        regex_string = r"("
        for token in tokens:
            processed_token = token
            for special_character in "()[]":
                processed_token = processed_token.replace(
                    special_character, f"\\{special_character}"
                )
            regex_string += processed_token + r"|"
        for token in regex_tokens:
            regex_string += token + r"|"
        regex_string += r".)"

        return re.compile(regex_string)

    def _check_encoding_type(self, encoding_type: str) -> None:
        if encoding_type not in {"one hot", "index"}:
            raise ValueError(
                f"unknown choice of encoding: {encoding_type}, muse be either 'one hot' or 'index'"
            )

    def _state_properties(self) -> Dict[str, Any]:
        dict_ = {"regex": self._re.pattern if self._re else ""}
        dict_["special_tokens"] = {
            name: val for name, val in self.special_tokens.items()
        }
        return dict_

    def _update_state(self, dict_: Dict[str, Any]) -> None:
        """Update the internal state with properties loaded from disc"""
        if dict_["regex"]:
            self._re = re.compile(dict_["regex"])
        else:
            self._re = None
        self._beginning_of_smiles_token = dict_["special_tokens"]["start"]
        self._end_of_smiles_token = dict_["special_tokens"]["end"]
        self._padding_token = dict_["special_tokens"]["pad"]
        self._unknown_token = dict_["special_tokens"]["unknown"]
        self._regex_tokens = []
        self._tokens = []


class SMILESAtomTokenizer(SMILESTokenizer):
    """A subclass of the `SMILESTokenizer` that treats all atoms as tokens.
    This tokenizer works by applying two different sets of regular expressions:
    one for atoms inside blocks ([]) and another for all other cases. This allows
    the tokenizer to find all atoms as blocks without having a comprehensive list
    of all atoms in the token list.
    """

    def __init__(
        self,
        *args,
        tokens: List[str] = None,
        smiles: List[str] = None,
        regex_tokens_patterns: List[str] = None,
        **kwargs,
    ) -> None:
        regex_tokens_patterns = regex_tokens_patterns or []

        smiles = smiles or []

        with warnings.catch_warnings(record=smiles != []):
            super().__init__(*args, **kwargs)
            super().add_tokens(["Br", "Cl"])
            super().add_regex_token_patterns(regex_tokens_patterns + [r"\[[^\]]*\]"])
        self.re_block_atom = re.compile(r"(Zn|Sn|Sc|[A-Z][a-z]?(?<!c|n|o|p|s)|se|as|.)")

        super().create_vocabulary_from_smiles(smiles)

    def tokenize(self, smiles: List[str]) -> List[List[str]]:
        """Converts a list of SMILES into a list of lists of tokens, where all atoms are
        considered to be tokens.
        The function first scans the SMILES for atoms and bracketed expressions
        uisng regular expressions. These bracketed expressions are then parsed
        again using a different regular expression.
        :param smiles: List of SMILES.
        :return: List of tokenized SMILES.
        """
        data_tokenized = super().tokenize(smiles)
        final_data = []
        for tokens in data_tokenized:
            final_tokens = []
            for token in tokens:
                if token.startswith("["):
                    final_tokens += self.re_block_atom.findall(token)
                else:
                    final_tokens.append(token)
            final_data.append(final_tokens)

        return final_data

### other function
def atomwise_tokenizer(smi, exclusive_tokens = None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens

def split_reaction(rct,mode='r'):
  #rct: the reation string
  #mode: 'm' seperate into moleculs/'r': seperate into reactant 
  #and product
  if mode=='m':
    R_P=rct.split('>>')
    assert len(R_P)==2
    product=R_P[1]
    try:
      first,second=R_P[0].split('.')
    except:
      #print(f'Only one reactant involved {rct}')
      return R_P[0],None,product
    
    return first,second,product
  else:
    R_P=rct.split('>>')
    assert len(R_P)==2
    return R_P[0],R_P[1]
  
def unpack_tuple(tmp: list):
  #[(),()...]
  m_list=[]
  for t in tmp:
    for item in t:
      m_list.append(item)
  
  return m_list

class Tokenizer(object):
  def __init__(self,data):
    #data is a list of molecus
    unique_char=set()
    unique_char.add('<sos>')
    for mol in data:
      tokens=atomwise_tokenizer(mol)
      for token in tokens:
        unique_char.add(token)
    
    unique_char.add('<eos>')
    
    unique_char=list(unique_char)
    self.mapping={'<pad>':0}
    for i, c in enumerate(unique_char,start=1):
      self.mapping[c]=i
    self.inv_mapping={c:i for i,c in self.mapping.items()}

    self.start_token=self.mapping['<sos>']
    self.end_token=self.mapping['<eos>']
    self.vocab_size=len(self.mapping)

  def encode_smile(self, mol, add_sos=False,add_eos=False):
    mol=atomwise_tokenizer(mol)
    out = [self.mapping[i] for i in mol]

    if add_eos:
        out = out + [self.end_token]  
    if add_sos:
      out = [self.start_token]+out     
    return torch.LongTensor(out)

  def batch_tokenize(self, batch, dec_in=False, tgt=False):
      if dec_in:
        out = map(lambda x: self.encode_smile(x,add_sos=True), batch)
      elif tgt:
        out = map(lambda x: self.encode_smile(x,add_eos=True), batch)
      else:
        out = map(lambda x: self.encode_smile(x), batch)

      return torch.nn.utils.rnn.pad_sequence(list(out), batch_first=True)
  
  def get_token_num(self):
    return len(self.mapping)
  
  def get_token_encoding(self,token):
    return self.mapping[token]

  def get_inv_mapping(self,x):
    out=[[]]
    for i,row in enumerate(x):
      out.append([])
      for keys in row:
        for key in keys:
          out[i].append(self.inv_mapping[int(key)])
    return out

  
def split_dataset(data,p_valid,p_test):
  assert(p_valid+p_test<1)
  train_size=round(len(data)*(1-p_valid-p_test))
  valid_size=round(len(data)*p_valid)
  test_size=round(len(data)*p_test)
  train_set,valid_set,test_set=random_split(dataset=data,lengths=[train_size,valid_size,test_size])
  return train_set,valid_set,test_set

  
  

