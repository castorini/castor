import argparse
from random import uniform
from subprocess import call


def get_param():
    learning_rate = round(10 ** uniform(-5, -3), 6)
    eps = round(10 ** uniform(-7, -3), 8)
    reg = round(10 ** uniform(-4, 0), 7)
    return learning_rate, eps, reg

def run(device, count, epochs):
    for _ in range(count):
        learning_rate, eps, reg = get_param()

        filename = "grid_trecqa_lr_{learning_rate}_eps_{eps}_reg_{reg}_device_{dev}.txt".format(learning_rate=learning_rate, eps=eps, reg=reg, dev=device)
        model_name = filename[:-4] + ".castor"
        command = "python -u main.py saved_models/{model} --epochs {epo} --device {dev} --dataset trecqa --batch-size 64 " \
                  "--lr {learning_rate} --epsilon {eps} --regularization {reg} --tensorboard --run-label {label} --dev_log_interval 100 " \
                  .format(epo=epochs, model=model_name, dev=device, learning_rate=learning_rate, eps=eps, reg=reg, label=filename)

        print("Running: " + command)
        with open(filename, 'w') as outfile:
            call(command, shell=True, stderr=outfile)

def run_wiki(device, count, epochs):
    for _ in range(count):
        learning_rate, eps, reg = get_param()

        filename = "grid_wikiqa_lr_{learning_rate}_eps_{eps}_reg_{reg}_device_{dev}.txt".format(learning_rate=learning_rate, eps=eps, reg=reg, dev=device)
        model_name = filename[:-4] + ".castor"
        command = "python -u main.py saved_models/{model} --epochs {epo} --device {dev} --dataset wikiqa --batch-size 64 " \
                  "--lr {learning_rate} --epsilon {eps} --regularization {reg} --tensorboard --run-label {label} --dev_log_interval 100 " \
                  "--dev_log_interval 100".format(epo=epochs, model=model_name, dev=device, learning_rate=learning_rate, eps=eps, reg=reg, label=filename)

        print("Running: " + command)
        with open(filename, 'w') as outfile:
            call(command, shell=True, stderr=outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyper parameters sweeper')
    parser.add_argument('--device', type=int, default=2, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--count', type=int, default=30, help='number of times to run')
    parser.add_argument('--epochs', type=int, default=8, help='number of epochs to run')
    args = parser.parse_args()
    run_wiki(args.device, args.count, args.epochs)