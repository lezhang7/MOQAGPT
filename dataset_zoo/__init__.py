## this file featch dataset from DATASET_DIR


from .mmcoqa import MmcoqaDataset
from .mmqa import MMQA
def get_dataset(dataset_name):
    dataset_name=dataset_name.lower()
    if dataset_name.lower()=="mmcoqa":
        return MmcoqaDataset("test")
    if dataset_name.lower()=="mmqa":
        return MMQA("dev")

