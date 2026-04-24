import argparse

parser = argparse.ArgumentParser(description='Data_Handler')

parser.add_argument('--dataset', type=str, default='path.csv')

args = parser.parse_args()
print(args)