import numpy as np
import random

def sample(softmax, temperature):
  EPSILON = 1e-15
  probs = (np.array(softmax) + EPSILON).astype('float64')
  probs = np.log(probs) / temperature
  probs = np.exp(probs)
  probs = probs / np.sum(probs)
  return np.random.choice(range(len(probs)), p=probs)

#TODO: saving/loading
#TODO: regularizers
class ThoughtHelper():
  def __init__(self, rnn, buffer_size=10000):
    self.trajectories = {}
    self.buffer_size = buffer_size
    self.buffer = []
    self.rnn = rnn

  def generate(self, trajectory, max_len=1000, temp_state=None,
    temperature=0.1):
    """predict from current trajectory onwards"""
    state = self.get_trajectory(trajectory) if temp_state is None else temp_state

    #generate some text
    gen = []
    for i in range(max_len):
      char = self.rnn.decode(state, return_probs=True)[0]
      char = sample(char, temperature) if temperature is not None else np.argmax(char)
      state = self.rnn.encode(char, state)[0]

      if char == 0:
        break
      gen.append(char)

    gen = bytes(gen).decode("utf-8", "ignore")
    return gen

  def remember(self, char, state, next_char):
    self.buffer.append((char, state, next_char))

    while len(self.buffer) > self.buffer_size:
      self.buffer.pop(0)

  def update(self, trajectory, text, add_to_buffer=True,
    add_null_terminator=True, zero_injection_rate=0.001):
    """updates state with text"""
    if add_null_terminator:
      text = text + "\0"

    #convert to list of bytes
    #need to add \0 to the front to learn the transition between null and first char
    #this assumes that the last character seen was actually \0
    text = list(bytes("\0"+text, "utf-8"))
    #grab trajectory state
    state = self.get_trajectory(trajectory)
    #for every char
    for i in range(len(text) - 1):
      char = text[i]
      next_char = text[(i + 1) % len(text)]

      if add_to_buffer:
        self.remember(char, state, next_char)
        #inject zero states (for resuming when states are lost)
        if np.random.random() < zero_injection_rate:
          self.remember(char, np.zeros([self.rnn.state_size]), next_char)

      #encode that char into the trajectory state
      state = self.rnn.encode(char, state)[0]

    #TODO: update state with \0?
    #yes.
    state = self.rnn.encode(next_char, state)[0]

    #update trajectory state
    self.reset_trajectory(trajectory, state)

  def reset_trajectory(self, trajectory, state=None):
    state = np.zeros([self.rnn.state_size]) if state is None else state
    #print(state)
    self.trajectories[trajectory] = state

  def get_trajectory(self, trajectory):
    return self.trajectories[trajectory]

  def get_batch(self, size):
    batch = [random.choice(self.buffer) for _ in range(size)]
    return zip(*batch)

  def train(self, batch_size=64, num_batches=1):
    for i in range(num_batches):
      chars, states, next_chars = self.get_batch(batch_size)
      loss = self.rnn.train_on_batch(chars, states, next_chars)

    return loss
