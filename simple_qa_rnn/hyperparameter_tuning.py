from random import randint, uniform
from subprocess import call

epochs = 10
count = 500
for id in range(count):
    learning_rate = 10 ** uniform(-6, 0)
    d_hidden = randint(100, 1000)
    n_layers = randint(1, 3)
    dropout = uniform(0.2, 0.6)
    clip = uniform(0.2, 0.6)

    command = "python train.py --cuda --batch_size 128 --lr {} --d_hidden {} --n_layers {} " \
              "--dropout_prob {} --clip {} >> results.txt".format(learning_rate, d_hidden, n_layers, dropout, clip)

    print("Running: " + command)
    call(command, shell=True)