import argparse

parser = argparse.ArgumentParser(description='Model_Training')

parser.add_argument('--dataset', type=str, default='path.csv')

args = parser.parse_args()
print(args)