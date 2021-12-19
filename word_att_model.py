import torch
import torch.nn as nn
import torch.nn.functional as F


class WordAttNet(nn.Module):
    def __init__(self, dict, hidden_size=50):
        super(WordAttNet, self).__init__()
        self.lookup = nn.Embedding(num_embeddings=dict.shape[0], embedding_dim=dict.shape[1]).from_pretrained(dict)
        self.lstm1 = nn.LSTM(dict.shape[1], hidden_size, bidirectional=True)
        self.fc1 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.fc3 = nn.Linear(2 * hidden_size, 1, bias=False)
        self.dropout1 = nn.Dropout(p=0.2)

    def forward(self, input):
        output = self.lookup(input)
        output1 = self.dropout1(output)
        f_output1, _ = self.lstm1(output1.float())
        weight = F.tanh(self.fc1(f_output1))
        weight = self.fc3(weight)  # seq * batch * 1
        weight = F.softmax(weight, 0)
        weight = weight * f_output1
        output1 = weight.sum(0).unsqueeze(0)
        return output1
    # output shape: 1 *batch *hidden_size


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
