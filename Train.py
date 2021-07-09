import argparse

parser = argparse.ArgumentParser("Experimental Setting")

parser.add_argument('--Cancer type', type = str, default = 'COAD',
                    help = 'Only including COAD, READ and CRC')
parser.add_argument('--level', type=str, default = 'slide',
                    help = 'Prediction level, slide or patient')
parser.add_argument('--hidden_dim', type = int, default = 2048,
                    help = 'patch features dimension')
parser.add_argument('--encoder_layer', type = int, default = 1,
                    help='Number of Transformer Encoder layer')
parser.add_argument('k_sample', type = int, default = 2,
                    help='Number of top and bottom cluster to be selected')

parser.add_argument('lr', type = float, default = 3e-4)
parser.add_argument('epoch', type = int, default = 60)
parser.add_argument('tau', type = float, default = 0.7)
parser.add_argument('kfold', type = int, default = 5)

if __name__ == '__main__':
    args = parser.parse_args()

