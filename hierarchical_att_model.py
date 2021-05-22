import torch
import torch.nn as nn
from sent_att_model import SentAttNet
from word_att_model import WordAttNet

class HierAttNet(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes,dict,
                 max_sent_length, max_word_length):
        super(HierAttNet, self).__init__()
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.word_att_net = WordAttNet( dict,word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)

    def forward(self, input):
        output_list1 = []
        input = input.permute(1, 0, 2) # 32 * 75 * 30
        for i in input:
            output1 = self.word_att_net(i.permute(1, 0))
            output_list1.append(output1)
        output = torch.cat(output_list1, 0)
        output1,output2,output3 = self.sent_att_net(output) # seq_len * batch * class

        return output1,output2,output3


    def loss_pre(self, pred_y, pred_l,pred_el,true_label):

        length = true_label[:,-1,-1].tolist()
        #print('shape is ',true_label[:,:,0].shape,pred_y.shape)    # 32 * 75 , 75 * 32 * 4
        #print(length)
        #print(pred_y.shape,pred_l.shape,pred_el.shape,true_label[:,:,0].shape,true_label[:,:,1].shape,true_label[:,:,2].shape)
        a = nn.CrossEntropyLoss()
        packed_y = torch.nn.utils.rnn.pack_padded_sequence(pred_y, length,enforce_sorted=False).data
        target_y = torch.nn.utils.rnn.pack_padded_sequence(true_label[:,:,0].permute(1,0), length,enforce_sorted=False).data
        loss_y  = a(packed_y, target_y)

        packed_l = torch.nn.utils.rnn.pack_padded_sequence(pred_l, length,enforce_sorted=False).data
        target_l = torch.nn.utils.rnn.pack_padded_sequence(true_label[:,:,1].permute(1,0), length,enforce_sorted=False).data
        loss_l  = a(packed_l, target_l)

        #print(true_label[:,:,0].shape,pred_y.shape)
        packed_el = torch.nn.utils.rnn.pack_padded_sequence(pred_el,length,enforce_sorted=False).data
        target_el = torch.nn.utils.rnn.pack_padded_sequence(true_label[:,:,2].permute(1,0), length,enforce_sorted=False).data
        loss_el  = a(packed_el, target_el)
        loss = loss_y + loss_l + loss_el

        return loss
