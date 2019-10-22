import numpy as np
from keras.layers import Dense, Input, Embedding, Concatenate, Flatten, BatchNormalization
from keras.models import Model
from keras.utils import to_categorical

def expand_if(arr, dim, axis=0):
  arr = np.array(arr)
  if len(arr.shape) == dim:
    arr = np.expand_dims(arr, axis=0)
  return arr

class ThoughtRNN():
  def create_model(embedding_size=8, state_size=32, state_acti="tanh",
    encoder_sizes=[64], decoder_sizes=[], hidden_acti="relu", use_bn=True):
    #build encoder
    char_input = Input([1])
    enc_state_input = Input([state_size])
    x = char_input
    x = Embedding(256, embedding_size)(x)
    x = Flatten()(x)
    x = Concatenate()([x, enc_state_input])
    for size in encoder_sizes:
      x = Dense(size, activation=hidden_acti)(x)
      if use_bn:
        x = BatchNormalization()(x)
    x = Dense(state_size, activation=state_acti)(x)

    encoder = Model(inputs=[char_input, enc_state_input], outputs=x)

    #build decoder
    dec_state_input = Input([state_size])
    x = dec_state_input
    for size in decoder_sizes:
      x = Dense(size, activation=hidden_acti)(x)
      if use_bn:
        x = BatchNormalization()(x)
    x = Dense(256, activation="softmax")(x)

    decoder = Model(inputs=dec_state_input, outputs=x)

    #build full model
    model_output = decoder(encoder([char_input, enc_state_input]))
    model = Model(inputs=[char_input, enc_state_input], outputs=model_output)

    return model, encoder, decoder

  def __init__(self, model):
    self.model = model
    #seems legit
    self.encoder = self.model.layers[-2]
    self.encoder = Model(self.encoder.inputs, self.encoder.outputs)
    self.decoder = self.model.layers[-1]
    self.decoder = Model(self.decoder.inputs, self.decoder.outputs)
    self.state_size = self.decoder.input_shape[-1]

  def encode(self, char, state):
    """takes a character and current state, returns new state.
    this can be called multiple times in a row to summarize a sequence
    """
    char = expand_if(char, 0, axis=1)
    state = expand_if(state, 1)

    return self.encoder.predict([char, state])

  def decode(self, state, return_probs=False):
    """decodes a state into the next character"""
    state = expand_if(state, 1)
    char = self.decoder.predict(state)
    if not return_probs:
      #not sure why this has to be axis=1....
      #char = np.argmax(char, axis=1)
      char = [np.random.choice(len(c), p=c) for c in char]

    return char

  def predict(self, char, state=None, return_probs=False):
    """runs full cycle encode -> decode, returns next char and new state"""
    state = np.zeros(self.state_size) if state is None else state
    state = self.encode(char, state)
    char = self.decode(state, return_probs=return_probs)
    return char, state

  def train_on_batch(self, chars, states, next_chars):
    """chars=byte values"""
    chars = np.array(chars)
    states = np.array(states)
    next_chars = to_categorical(next_chars, 256)
    return self.model.train_on_batch([chars, states], next_chars)
