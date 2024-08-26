from dataset import WeatherDataset
from models.adaptive_normalizer import AdaptiveNormalizer


class BatchGenerator:

    def __init__(self, train_data, val_data, test_data, normalize_flag, params):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.dataset_params = params
        self.normalize_flag = normalize_flag

        if self.normalize_flag:
            self.normalizer = AdaptiveNormalizer(output_dim=params['output_dim'])
        else:
            self.normalizer = None

        self.weather_dict = self.__split_data(self.train_data, self.val_data, self.test_data)
        self.dataset_dict = self.__create_sets()

    def __split_data(self, train_data, val_data, test_data):

        data_dict = {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }

        return data_dict

    def __create_sets(self):
        hurricane_dataset = {}
        for i in ['train', 'val', 'test']:
            dataset = WeatherDataset(weather_data=self.weather_dict[i],
                                     normalizer=self.normalizer,
                                     **self.dataset_params)
            hurricane_dataset[i] = dataset

        return hurricane_dataset

    def num_iter(self, dataset_name):
        return self.dataset_dict[dataset_name].num_iter

    def generate(self, dataset_name):
        selected_loader = self.dataset_dict[dataset_name]
        yield from selected_loader.next()
