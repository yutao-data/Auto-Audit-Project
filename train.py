#!/usr/bin/env python3

import torch
import pandas
import numpy
import os
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    parser = argparse.ArgumentParser(
        "Train a model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--epochs', type=int, default=3900)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--log_level', type=str, default='INFO')
    parser.add_argument('--show_loss_history', action='store_false')
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')

    logging.info('Using device: {}'.format(device))

    logging.info('Loading data...')
    dataset = DataSet()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True)

    logging.info('Creating model...')
    model = Model(dataset.input_size)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # auto learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=100, verbose=True)
    weight = torch.tensor([1/dataset.weight_dict[i]
                          for i in range(len(dataset.weight_dict))])
    weight = weight.to(device)
    print('weight:', weight)
    loss_func = torch.nn.CrossEntropyLoss(weight=weight)

    history = []
    epoch_num = args.epochs

    pbar = tqdm(range(epoch_num))
    for epoch in pbar:
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            # print('labels:', labels)
            outputs = model(inputs)
            pred = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)
            # print('  pred:', pred)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        pbar.set_description('Train loss: %.8f' % numpy.mean(history[-100:]))
        scheduler.step(numpy.mean(history[-100:]))

    # save model
    torch.save(model.state_dict(), 'model.pth')

    # load model
    # model.load_state_dict(torch.load('model.pth'))

    # test
    test_dataset = DataSet(test=True, test_size=args.test_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=16, shuffle=True)
    test_loss = 0
    test_correct = 0
    total_count = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            pred = torch.nn.functional.softmax(outputs, dim=1).argmax(dim=1)
            loss = loss_func(outputs, labels)
            test_loss += loss.item()
            test_correct += torch.sum(pred == labels).item()
            total_count += len(labels)
            print('labels:', labels)
            print('  pred:', pred)
            print('----------------------------------------')
        print('Test Accuracy: {}'.format(test_correct / total_count))

    if args.show_loss_history:
        plt.plot(history)
        plt.show()


# simple neural network model
class Model(torch.nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 2)

    def forward(self, x):
        x = self.fc1(x)

        return x


class DataSet(torch.utils.data.Dataset):
    def __init__(self, test=False, test_size=0.2):
        self.source = torch.tensor(pandas.read_csv(
            'source.csv', index_col='id').values).float()
        self.target = torch.tensor(pandas.read_csv(
            'target.csv', index_col='id')[['target_roa']].values).float()
        print('DataSet loaded', self.source.shape, self.target.shape)

        self.input_size = self.source.shape[1]

        luckly_number = random.random()
        random.Random(luckly_number).shuffle(self.source)
        random.Random(luckly_number).shuffle(self.target)

        if test:
            self.source = self.source[:int(len(self.source) * test_size)]
            self.target = self.target[:int(len(self.target) * test_size)]
        else:
            self.source = self.source[int(len(self.source) * test_size):]
            self.target = self.target[int(len(self.target) * test_size):]

        self.weight_dict = {}
        for i in range(len(self.source)):
            target_type = 0 if self.target[i] > 0 else 1
            if target_type not in self.weight_dict:
                self.weight_dict[target_type] = 1
            else:
                self.weight_dict[target_type] += 1
        print(self.weight_dict)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        source = self.source[index]
        target = self.target[index]

        if target > 0:
            target = 0
        else:
            target = 1

        # normalization
        source = (source - source.mean()) / source.std()
        return source, target


def test():
    dataset = DataSet()
    for i in range(len(dataset)):
        source, target = dataset[i]
        if numpy.isnan(source).any() or numpy.isnan(target).any():
            print('NaN found')
            print(i, source, target)
            continue


if __name__ == '__main__':
    main()
