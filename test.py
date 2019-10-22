import random
import numpy as np
from keras.optimizers import Adam
from thoughtrnn import ThoughtRNN, ThoughtHelper

#create rnn model
hidden_acti = "relu" #"tanh"
encoder_sizes = [256]
decoder_sizes = [256]
state_size = 32
embedding_size = 8
model, enc, dec = ThoughtRNN.create_model(encoder_sizes=encoder_sizes,
  decoder_sizes=decoder_sizes, state_size=state_size,
  embedding_size=embedding_size, hidden_acti=hidden_acti)

model.compile(
  loss="categorical_crossentropy",
  optimizer=Adam(1e-4) #1e-3 works ok. 1e-4 even better
)

model.summary()

zero_injection_rate = 0.001
batch_size = 64
buffer_size = 10000
gen_len = 250
temperature = 0.1
trajectory_name = "foo"

#create rnn and helper
rnn = ThoughtRNN(model)
helper = ThoughtHelper(rnn, buffer_size=buffer_size)
#create trajectory
helper.reset_trajectory(trajectory_name)

texts = [
  #"aa",
  #"Hello World!",
  #"0123456789",
  #"9876543210",
  "Never gonna give you up",
  "Never gonna let you down",
  "Never gonna run around and desert you",
  "Never gonna make you cry",
  "Never gonna say goodbye",
  "Never gonna tell a lie and hurt you",
]

steps = 0
phrase_steps = 50
gen_steps = 250
batches_per_step = 1

while True:
  if steps % phrase_steps == 0:
    text = random.choice(texts)
    #add text to buffer
    helper.update(trajectory_name, text, zero_injection_rate=zero_injection_rate)

  #train
  loss = helper.train(batch_size, batches_per_step)

  steps += 1
  if (steps + 1) % gen_steps == 0:
    print(steps + 1, ":", loss, len(helper.buffer))
    #generate text
    #generated = helper.generate(trajectory_name, gen_len, temp_state=np.random.uniform(-1, 1, size=rnn.state_size))
    generated = helper.generate(trajectory_name, gen_len, temperature=temperature)
    #generated = helper.generate(trajectory_name, gen_len, temp_state=np.zeros([rnn.state_size]))
    print("<", generated, ">")
    print()
