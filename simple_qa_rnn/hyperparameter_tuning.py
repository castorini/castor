from random import randint, uniform
from subprocess import call

epochs = 20
count = 500
for id in range(count):
    learning_rate = 10 ** uniform(-7, -3)
    d_hidden = randint(300, 800)
    n_layers = randint(1, 4)
    dropout = uniform(0.15, 0.65)
    clip = uniform(0.4, 1)

    command = "python train.py --cuda --dev_every 500 --log_every 250 --batch_size 128 " \
                "--epochs {} --lr {} --d_hidden {} --n_layers {} --dropout_prob {} --clip {} >> " \
                    "results.txt".format(epochs, learning_rate, d_hidden, n_layers, dropout, clip)

    print("Running: " + command)
    call(command, shell=True)
