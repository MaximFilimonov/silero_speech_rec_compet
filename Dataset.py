import torch

class Dataset(torch.utils.data.Dataset):
    '''
    Characterizes a dataset for PyTorch
    '''
    
    def __init__(self, list_IDs, labels):

        self.labels = labels
        self.list_IDs = list_IDs


    def __len__(self):

        return len(self.list_IDs)

    
    def __getitem__(self, index):

        ID = self.list_IDs[index]

        X = torch.load('data/' + ID + '.pt')
        y = self.labels[ID]

        return X, y