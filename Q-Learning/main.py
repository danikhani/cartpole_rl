import argparse
import train
import test

def parse_input():
    parser = argparse.ArgumentParser(description='Inverses Pendel mit QLearning')
    parser.add_argument('-t,','--training',action='store_true',help='Switches to training mode. Default=false')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    number_of_buckets = (5, 1, 6, 3)
    args = parse_input()
    if args.training is True:
        print('training started')
        train.start_train()
    else:
        print('test started')
        test.start_test()