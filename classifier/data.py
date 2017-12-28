import torch
import os
from torch.utils.data import Dataset, DataLoader

class MathSolverDataset(Dataset):
    """Math Solver Dataset"""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.equation_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        print('len(self.equation_frame)', len(self.equation_frame))
        return len(self.equation_frame)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.root_dir, self.equation_frame.ix[idx, 0])
        #image = io.imread(img_name)
        #equation = self.equation_frame.ix[idx, 1:].as_matrix().astype('float')
        #sample = {'text': text, 'equation': equation}
        return {}
