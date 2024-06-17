from collections import Counter
import numpy as np
import math
import random

"""
CS6120 Homework 2 - starter code
"""

# constants
SENTENCE_BEGIN = "<s>"
SENTENCE_END = "</s>"
UNK = "<UNK>"


# UTILITY FUNCTIONS

def create_ngrams(tokens: list, n: int) -> list:
  """Creates n-grams for the given token sequence.
  Args:
    tokens (list): a list of tokens as strings
    n (int): the length of n-grams to create

  Returns:
    list: list of tuples of strings, each tuple being one of the individual n-grams
  """
  # STUDENTS IMPLEMENT 
  ngrams = []
  for i in range(len(tokens) - n + 1):
      ngram = tuple(tokens[i:i + n])
      ngrams.append(ngram)
  return ngrams
  pass

def read_file(path: str) -> list:
  """
  Reads the contents of a file in line by line.
  Args:
    path (str): the location of the file to read

  Returns:
    list: list of strings, the contents of the file
  """
  # PROVIDED
  f = open(path, "r", encoding="utf-8")
  contents = f.readlines()
  f.close()
  return contents

def tokenize_line(line: str, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize a single string. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    line (str): text to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - a single line tokenized
  """
  # PROVIDED
  inner_pieces = None
  if by_char:
    inner_pieces = list(line)
  else:
    inner_pieces = line.split()

  if ngram == 1:
    tokens = [sentence_begin] + inner_pieces + [sentence_end]
  else:
    tokens = ([sentence_begin] * (ngram - 1)) + inner_pieces + ([sentence_end] * (ngram - 1))
  return tokens


def tokenize(data: list, ngram: int, 
                   by_char: bool = True, 
                   sentence_begin: str=SENTENCE_BEGIN, 
                   sentence_end: str=SENTENCE_END):
  """
  Tokenize each line in a list of strings. Glue on the appropriate number of 
  sentence begin tokens and sentence end tokens (ngram - 1), except
  for the case when ngram == 1, when there will be one sentence begin
  and one sentence end token.
  Args:
    data (list): list of strings to tokenize
    ngram (int): ngram preparation number
    by_char (bool): default value True, if True, tokenize by character, if
      False, tokenize by whitespace
    sentence_begin (str): sentence begin token value
    sentence_end (str): sentence end token value

  Returns:
    list of strings - all lines tokenized as one large list
  """
  # PROVIDED
  total = []
  for line in data:
    line = line.strip()
    if len(line) == 0:
      continue
    tokens = tokenize_line(line, ngram, by_char, sentence_begin, sentence_end)
    total += tokens
  return total


class LanguageModel:

  def __init__(self, n_gram):
    """Initializes an untrained LanguageModel
    Args:
      n_gram (int): the n-gram order of the language model to create
    """
    # STUDENTS IMPLEMENT
    self.n_gram = n_gram
    self.token_counts = Counter()
    self.ngram_counts = Counter()
    self.context_counts = Counter()
    self.vocab = set()

  
  def train(self, tokens: list, verbose: bool = False) -> None:
    """Trains the language model on the given data. Assumes that the given data
    has tokens that are white-space separated, has one sentence per line, and
    that the sentences begin with <s> and end with </s>
    Args:
      tokens (list): tokenized data to be trained on as a single list
      verbose (bool): default value False, to be used to turn on/off debugging prints
    """
    self.token_counts = Counter(tokens)
    tokens_list = [UNK if self.token_counts[word] == 1 else word for word in tokens]
    
    for i in range(len(tokens_list) - self.n_gram + 1):
      ngram = tuple(tokens_list[i:i + self.n_gram])
      self.ngram_counts[ngram] += 1
      context = ngram[:-1]
      self.context_counts[context] += 1
      self.vocab.add(ngram[-1])
      self.vocab_size = len(self.vocab)
  
  def score(self, sentence_tokens: list) -> float:
    """Calculates the probability score for a given string representing a single sequence of tokens.
    Args:
      sentence_tokens (list): a tokenized sequence to be scored by this model
 
    Returns:
      float: the probability value of the given tokens for this model
    """
    total_prob = 1
    sentence_tokens = [token if token in self.vocab else UNK for token in sentence_tokens]
    ngram_tokens = create_ngrams(sentence_tokens, self.n_gram)
    ngram_count = 0
    ngram_prob = 0
 
    if self.n_gram==1:
      for ngram in ngram_tokens:
          ngram_count = self.ngram_counts.get(ngram, 0)
          if ngram_count == 0: ngram_count += 1
          context = tuple(ngram[:-1])
          ngram_prob = (ngram_count + 1) / (self.context_counts[context] + self.vocab_size)
          total_prob *= ngram_prob
          
    else:
      for ngram in ngram_tokens:
          ngram_count = self.ngram_counts.get(ngram, 0)
          context_count = self.context_counts.get(ngram[:-1], 0)
          ngram_prob = (ngram_count + 1) / (context_count + self.vocab_size)
          total_prob *= ngram_prob
    return total_prob  

  def generate_sentence(self) -> list:
    """Generates a single sentence from a trained language model using the Shannon technique.
      
    Returns:
      list: the generated sentence as a list of tokens
    """
    # STUDENTS IMPLEMENT
    current_context = [SENTENCE_BEGIN] * (self.n_gram - 1)
    sentence = [SENTENCE_BEGIN]
    total_prob = 0
    my_vocab = self.vocab
    words = []
    probab = []

    for word in my_vocab:
      if not word == '<s>':
        ngram = tuple(current_context + [word])
        # Calculate the probability of the word
        prob = self.score(ngram)
        total_prob += prob
        words.append(word)
        probab.append(prob)

    while True:
      chosen_word = random.choices(words,weights = probab, k=1)[0]
      if chosen_word == SENTENCE_END:
        break

      sentence.append(chosen_word)
      if len(sentence) >= 10: 
        break
      current_context = (current_context + [chosen_word])[1:]
    return sentence

  def generate(self, n: int) -> list:
    """Generates n sentences from a trained language model using the Shannon technique.
    Args:
      n (int): the number of sentences to generate
      
    Returns:
      list: a list containing lists of strings, one per generated sentence
    """
    # PROVIDED
    return [self.generate_sentence() for i in range(n)]


  def perplexity(self, sequence: list) -> float:
    """Calculates the perplexity score for a given sequence of tokens.
    Args:
      sequence (list): a tokenized sequence to be evaluated for perplexity by this model
      
    Returns:
      float: the perplexity value of the given sequence for this model
    """
    log_prob = self.score(sequence) 
    seq_count = Counter(sequence)
    num_start_tok = seq_count[SENTENCE_BEGIN]
    num_tokens = len(sequence) - num_start_tok
    perplexity = log_prob**(-1/num_tokens)
    return perplexity
  
# not required
if __name__ == '__main__':
  print()
  print("if having a main is helpful to you, do whatever you want here, but please don't produce too much output :)")
  print("call a function")
  print(tokenize_line("tokenize this sentence!", 2, by_char=False))
  print(tokenize(["apples are fruit", "bananas are too"], 2, by_char=False))