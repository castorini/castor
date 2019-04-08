from argparse import ArgumentParser
import os

def get_args():
    parser = ArgumentParser(description="SM CNN")

    parser.add_argument('model_outfile', help='file to save final model')
    parser.add_argument('--dataset', type=str, help='trecqa|wikiqa', default='trecqa')
    parser.add_argument('--no_cuda', action='store_false', help='do not use cuda', dest='cuda')
    parser.add_argument('--word-vectors-dir', help='word vectors directory',
                        default=os.path.join(os.pardir, 'Castor-data', 'embeddings', 'word2vec'))
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='aquaint+wiki.txt.gz.ndim=50.txt')
    parser.add_argument('--word-vectors-dim', type=int, default=50,
                        help='number of dimensions of word vectors (default: 50)')
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=3435)
    parser.add_argument('--resume_snapshot', type=str, default=None)
    parser.add_argument('--dev_every', type=int, default=30)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--output_channel', type=int, default=100)
    parser.add_argument('--filter_width', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epoch_decay', type=int, default=15)
    parser.add_argument('--vector_cache', type=str, default='data/word2vec.trecqa.pt')
    parser.add_argument('--trained_model', type=str, default="")
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--onnx', action='store_true', help='export model to onnx')
    parser.add_argument('--mode', type=str, default='rand')
    parser.add_argument('--keep-results', action='store_true',
                        help='store the output score and qrel files into disk for the test set')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--run-label', type=str, help='label to describe run')


    args = parser.parse_args()
    return args
