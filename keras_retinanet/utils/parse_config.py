import json


def parse_config(config_path):
    json_data_file = open(config_path)
    configs = json.load(json_data_file)
    json_data_file.close()
    # DatasetConfigs = configs['Dataset']
    # TrainConfigs = configs['Train']
    # DataAugConfigs = configs['Data_Augmentation']
    return configs
