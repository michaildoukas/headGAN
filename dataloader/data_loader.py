from dataloader.custom_dataset_data_loader import CustomDatasetDataLoader

def CreateDataLoader(opt, start_idx=0):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt, start_idx)
    return data_loader
