import importlib
from torch.utils.data import DataLoader


def getDataLoaders(opt) :
    # Select Dataset Name
    if opt.dataType == "w2s" :
        datasetName = "w2sDataset"
        imageSavePath = f"{opt.resultsDir}/{opt.name}/{opt.dataType}/sigma-{opt.sigma}"
    elif opt.dataType == "Hagen" :
        datasetName = "HagenDataset"
        imageSavePath = f"{opt.resultsDir}/{opt.name}/{opt.dataType}"
    else :
        raise NotImplementedError(f"{opt.dataType} is not supported")

    # Import Python Code
    fileName = importlib.import_module(f"data.{datasetName}")
    
    # Create Dataset Instance
    dataset = fileName.__dict__[datasetName](opt)
    
    # Train PyTorch DataLoader Instance
    dataLoader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            drop_last=False, 
                            num_workers=opt.numWorkers)
    
    
    return imageSavePath, dataLoader

