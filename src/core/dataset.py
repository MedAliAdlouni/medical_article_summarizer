import os
import pandas as pd
from torch.utils.data import Dataset

from .utils import extract_data


class ArticlesAndAbstracts(Dataset):
    def __init__(self, conf):
        super().__init__()

        self.path = conf["dataset"]["path"]
        self.subset = conf["dataset"]["subset"]
        self.data_path = os.path.join(self.path, self.subset)

        self.pdf_obs = extract_data(self.data_path, "OBS", num_rows=10)
        self.pdf_rct = extract_data(self.data_path, "RCT", num_rows=10)

        self.pdf = pd.concat([self.pdf_obs, self.pdf_rct], ignore_index=True)
        self.pdf["id"] = range(len(self.pdf))

    def __len__(self):
        return len(self.pdf)

    def __getitem__(self, index):
        sample = self.pdf.iloc[index, :]
        ind, article, abstract = sample["id"], sample["article"], sample["abstract"]
        return ind, article, abstract



