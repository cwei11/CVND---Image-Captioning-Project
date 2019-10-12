import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        # Inherit parent class
        
        super(DecoderRNN, self).__init__() 
        # Get all passed in para as in class value
       
        # Creating embedding layer to turn words into a vector
        self.emdededSys = nn.Embedding(vocab_size, embed_size)
        #Creating a LSTM system take embededword vectors and output hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        #Create the linear layer to map the hidden state of tags output dimension 
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)
        # init function from previous notbook. I think it is not needed to define extra method
        self.emdededSys.weight.data.uniform_(-0.1, 0.1)
        self.hidden2tag.weight.data.uniform_(-0.1, 0.1)
        self.hidden2tag.bias.data.fill_(0)
       
    
    def forward(self, features, captions):
        # Using embedding according to LSTM for Part for Speech Tagging
        embeddings = self.emdededSys(captions[:,: -1])
        # Cap 
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.hidden2tag(hiddens)
        return outputs
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass