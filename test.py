from cmrxrecon.dataloader.cmr_dataloader import CMR_Dataloader

if __name__ == '__main':
    dataset = CMR_Dataloader('/home/kadotab/projects/def-mchiew/kadotab/SingleCoil/Mapping/TrainingSet/FullSample')
    dataset[0]
