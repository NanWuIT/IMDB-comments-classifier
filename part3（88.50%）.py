import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as F
import torch.optim as topti
from torchtext import data
from torchtext.vocab import GloVe
from imdb_dataloader import IMDB

import re

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
  cleantext = re.sub(cleanr, '', raw_html)
  cleantext = re.sub(r"[^a-zA-Z ]", "", cleantext)
  return cleantext


# Class for creating the neural network.
class Network(tnn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.gru = tnn.GRU(input_size=50, hidden_size=180, batch_first=True, num_layers=2)
        self.fc1 = tnn.Linear(180, 128)
        self.fc3 = tnn.Linear(128, 128)
        # self.fc4 = tnn.Linear(128, 128)
        # self.fc5 = tnn.Linear(128, 128)
        # self.fc6 = tnn.Linear(128, 128)
        self.fc2 = tnn.Linear(128, 1)

    def forward(self, input, length):
        """
        DO NOT MODIFY FUNCTION SIGNATURE
        Create the forward pass through the network.
        """
        packedInput = torch.nn.utils.rnn.pack_padded_sequence(input=input, lengths=length, batch_first=True)
        packed_output, ht = self.gru(packedInput)

        # get the output of the last hidden state (batch_size, hidden_state_100) -> linear -> (batch_size, 64)
     
        out = self.fc1(ht[-1])
        out = tnn.functional.relu(out)

        out = self.fc3(out)
        out = tnn.functional.relu(out)
        # out = self.fc4(out)
        # out = tnn.functional.relu(out)
        # out = self.fc5(out)
        # out = tnn.functional.relu(out)
        # out = self.fc6(out)
        # out = tnn.functional.relu(out)

        # (batch_size, 64) -> linear -> (batch_size, 1)
        out = self.fc2(out)
        
        # flatten output (batch_size, 1) -> (batch_size)
        out = out.view(-1)
        return out


class PreProcessing():
    def pre(x):
        """Called after tokenization"""
        x_clean = cleanhtml(' '.join(x)).split()
        if len(x_clean) != 0:
            x = x_clean
        # x = list(np.random.permutation(x))
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


class PreProcessingAug():
    def pre(x):
        """Called after tokenization"""
        x_clean = cleanhtml(' '.join(x)).split()
        if len(x_clean) != 0:
            x = x_clean
        x = list(np.random.permutation(x))
        return x

    def post(batch, vocab):
        """Called after numericalization but prior to vectorization"""
        return batch

    text_field = data.Field(lower=True, include_lengths=True, batch_first=True, preprocessing=pre, postprocessing=post)


def lossFunc():
    """
    Define a loss function appropriate for the above networks that will
    add a sigmoid to the output and calculate the binary cross-entropy.
    """
    return torch.nn.BCEWithLogitsLoss()

def main():
    # Use a GPU if available, as it should be faster.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: " + str(device))

    # Load the training dataset, and create a data loader to generate a batch.
    textField = PreProcessing.text_field
    labelField = data.Field(sequential=False)

    train, dev = IMDB.splits(textField, labelField, train="train", validation="dev")

    textField.build_vocab(train, dev, vectors=GloVe(name="6B", dim=50))
    labelField.build_vocab(train, dev)

    trainLoader, testLoader = data.BucketIterator.splits((train, dev), shuffle=True, batch_size=64,
                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)
    
    trainLoaders = [trainLoader]
    textFields = [textField]

    # for i in range(3):
    #     textField0 = PreProcessingAug.text_field
    #     labelField0 = data.Field(sequential=False)

    #     train0, dev0 = IMDB.splits(textField0, labelField0, train="train", validation="dev")

    #     textField0.build_vocab(train0, dev0, vectors=GloVe(name="6B", dim=50))
    #     labelField0.build_vocab(train0, dev0)

    #     trainLoader0, testLoader0 = data.BucketIterator.splits((train0, dev0), shuffle=True, batch_size=64,
    #                                                         sort_key=lambda x: len(x.text), sort_within_batch=True)
    #     trainLoaders.append(trainLoader0)
    #     textFields.append(textField0)

    net = Network().to(device)
    criterion =lossFunc()
    optimiser = topti.Adam(net.parameters(), lr=0.001)  # Minimise the loss using the Adam algorithm.

    preAccuracies = [0]

    for epoch in range(8):
        running_loss = 0
        

        for j, tl in enumerate(trainLoaders):
          for i, batch in enumerate(tl):
              # for _ in range(3):
              # Get a batch and potentially send it to GPU memory.
              inputs, length, labels = textFields[j].vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                  device), batch.label.type(torch.FloatTensor).to(device)
              # print(inputs)

              labels -= 1

              # PyTorch calculates gradients by accumulating contributions to them (useful for
              # RNNs).  Hence we must manually set them to zero before calculating them.
              optimiser.zero_grad()

              # Forward pass through the network.
              output = net(inputs, length)

              loss = criterion(output, labels)

              # Calculate gradients.
              loss.backward()

              # Minimise the loss according to the gradient.
              optimiser.step()

              running_loss += loss.item()

              if i % 32 == 31:
                  print("Epoch: %2d, Batch: %4d, Loss: %.3f" % (epoch + 1, i + 1, running_loss / 32))
                  running_loss = 0

        with torch.no_grad():
            num_correct = 0
            for batch in testLoader:
                # Get a batch and potentially send it to GPU memory.
                inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                    device), batch.label.type(torch.FloatTensor).to(device)

                labels -= 1

                # Get predictions
                outputs = torch.sigmoid(net(inputs, length))
                predicted = torch.round(outputs)

                num_correct += torch.sum(labels == predicted).item()

            accuracy = 100 * num_correct / len(dev)
            if accuracy > max(preAccuracies):
                # Save mode
                torch.save(net.state_dict(), "./model.pth")
                print("Saved model")
            preAccuracies.append(accuracy)

            print(f"Classification accuracy: {accuracy}")

    num_correct = 0

    # Save mode
    # torch.save(net.state_dict(), "./model.pth")
    # print("Saved model")

    # Evaluate network on the test dataset.  We aren't calculating gradients, so disable autograd to speed up
    # computations and reduce memory usage.
    with torch.no_grad():
        for batch in testLoader:
            # Get a batch and potentially send it to GPU memory.
            inputs, length, labels = textField.vocab.vectors[batch.text[0]].to(device), batch.text[1].to(
                device), batch.label.type(torch.FloatTensor).to(device)

            labels -= 1

            # Get predictions
            outputs = torch.sigmoid(net(inputs, length))
            predicted = torch.round(outputs)

            num_correct += torch.sum(labels == predicted).item()

    accuracy = 100 * num_correct / len(dev)

    print(f"Classification accuracy: {accuracy}")

if __name__ == '__main__':
    main()
