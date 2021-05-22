import torch
import torch.nn as nn
import torch.nn.functional as F

class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=10):
        super(SentAttNet, self).__init__()
        self.lstm1 = nn.LSTM(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self.lstm2 = nn.LSTM(2 * word_hidden_size, sent_hidden_size, bidirectional=True)

        self.cnn1 = nn.Conv1d(400,128,3,padding=1)
        self.cnn2 = nn.Conv1d(400,128,5,padding=2)
        self.fc1 = nn.Linear(2 * sent_hidden_size, 4)
        self.fc2 = nn.Linear(2 * sent_hidden_size, 7)
        self.fc3 = nn.Linear(128 * 2, 19)
        self.fc4 = nn.Linear(128 * 2, 19)
        self.dropout = nn.Dropout(p=0.5)

        self.embedding = nn.Embedding(19, embedding_dim=256)
        self.cnn3 = nn.Conv1d(256,128,3,padding=1)
        self.cnn4 = nn.Conv1d(256,128,5,padding=2)

    def forward(self, input):
        # label:batch * seq_len * 4,input1 :seq_len * batch * ( 2  * hidden_size)
        input = self.dropout(input)

        encoder_output_1, encoder_hidden = self.lstm1(input)
        output1 = self.fc1(encoder_output_1)
        #embedding = self.embedding(torch.argmax(output1,2))
        encoder_output_2, encoder_hidden = self.lstm2(input)
        output2 = self.fc2(encoder_output_2)

        encoder_output_3 = self.cnn1(torch.cat((encoder_output_1,encoder_output_2),2).permute(1,2,0)).permute(2,0,1)
        encoder_output_4 = self.cnn2(torch.cat((encoder_output_1,encoder_output_2),2).permute(1,2,0)).permute(2,0,1)
        
        input = torch.cat((encoder_output_3,encoder_output_4),2)
        output3 = self.fc3(F.relu(input))
        
        embedding = self.embedding(output3.argmax(2))
        input = embedding + input
        
        encoder_output_3 = self.cnn3(input.permute(1,2,0)).permute(2,0,1)
        encoder_output_4 = self.cnn4(input.permute(1,2,0)).permute(2,0,1)
        input = torch.cat((encoder_output_3,encoder_output_4),2)
        output3 = self.fc4(F.relu(input))
        
        return output1,output2,output3

if __name__ == "__main__":
    abc = SentAttNet()
