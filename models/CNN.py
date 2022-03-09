import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, dimension, wordData, numberOfFilters, sizeOfFilters):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(wordData.vectors)
        
        self.conv1 = nn.Conv2d(1, numberOfFilters, kernel_size=(sizeOfFilters[0], dimension))
        self.conv2 = nn.Conv2d(1, numberOfFilters, kernel_size=(sizeOfFilters[1], dimension))
        self.fc1 = nn.Linear(dimension, 1)

    def forward(self, inputTensor):
        embedded = self.embedding(inputTensor)
        embedded = embedded.permute(1, 0, 2)
        embedded = torch.reshape(embedded, (embedded.shape[0], 1, embedded.shape[1], embedded.shape[2]))
        
        input1 = nn.functional.relu(self.conv1(embedded))
        max_pool1 = nn.MaxPool2d((input1.shape[2], 1), stride=1)
        input1 = max_pool1(input1)
        input1 = torch.reshape(input1, (input1.shape[0], input1.shape[1]))
        
        input2 = nn.functional.relu(self.conv2(embedded))
        max_pool2 = nn.MaxPool2d((input2.shape[2], 1), stride=1)
        input2 = max_pool2(input2)
        input2 = torch.reshape(input2, (input2.shape[0], input2.shape[1]))
        
        output = torch.cat((input1, input2), dim=1)
        outputTensor = self.fc1(output)
        return outputTensor