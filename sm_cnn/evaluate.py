import shlex
import subprocess

def evaluate(instances):
    sorted_instances = sorted(instances, key=lambda x: (x[0], -x[1]))
    with open('run.txt', 'w') as run, open('qrel.txt', 'w') as qrel:
        i = 0
        prev_qid = None
        for instance in sorted_instances:
            qid, predicted, gold = instance[0], instance[1], instance[2]

            if prev_qid != qid:
                i = 0

            # 32.1 0 1 0 0.13309887051582336 smmodel
            run.write('{} Q0 {} 0 {} sm_model\n'.format(qid, i, predicted))
            qrel.write('{} 0 {} {}\n'.format(qid, i, gold))

            prev_qid = qid

        pargs = shlex.split("./eval/trec_eval.9.0/trec_eval -m map -m recip_rank qrels.txt run.txt")
        p = subprocess.Popen(pargs, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pout, perr = p.communicate()

        lines = pout.split(b'\n')
        map = float(lines[0].strip().split()[-1])
        mrr = float(lines[1].strip().split()[-1])
        return map, mrr
