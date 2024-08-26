import torch
import numpy as np


class WeatherDataset:
    def __init__(self, weather_data, input_dim, output_dim,
                 window_in_len, window_out_len, batch_size, normalizer, shuffle,seed):

        self.weather_data = weather_data
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.window_in_len = window_in_len
        self.window_out_len = window_out_len
        self.total_window_len = window_in_len + window_out_len
        self.batch_size = batch_size
        self.num_iter = 0
        self.normalizer = normalizer
        self.shuffle = shuffle
        self.seed = seed

    def next(self):

        weather_data = self.__create_buffer(in_data=self.weather_data,seed=self.seed)
        self.num_iter = len(weather_data)
        prev_batch = None

        for i in range(self.num_iter):
            batch_data = torch.from_numpy(self.__load_batch(batch=weather_data[i]))

            if self.normalizer:
                batch_data = self.normalizer.norm(batch_data)

            # create x and y
            x = batch_data[:, :self.window_in_len, :,:]
            y = batch_data[:, self.window_in_len:, :,:]

            # create flow matrix
            if prev_batch is None:
                prev_batch = torch.zeros_like(batch_data)

            yield x, y

    def __create_buffer(self, in_data, seed):

        total_frame = len(in_data)

        all_data = []
        batch = []
        j = 0

        for i in range(total_frame-self.total_window_len):

            if j < self.batch_size:
                batch.append(in_data[i:i+self.total_window_len])
                j += 1
                if j == self.batch_size:
                    all_data.append(np.stack(batch, axis=0))
                    batch = []
                    j = 0

        if self.shuffle:
            all_data = np.stack(all_data)
            all_data = all_data.reshape(len(all_data)*self.batch_size, -1)
            random = np.random.RandomState(seed=seed).permutation(len(all_data))
            all_data = all_data[random]

            all_data = all_data.reshape(-1, self.batch_size, all_data.shape[-1])

        return all_data

    @staticmethod
    def __load_batch(batch):

        batch_size, win_len = batch.shape
        flatten_b = batch.flatten()

        list_arr = []
        for i in range(len(flatten_b)):
            list_arr.append(np.load(flatten_b[i]))

        return_batch = np.stack(list_arr, axis=0)
        other_dims = return_batch.shape[1:]
        return_batch = return_batch.reshape((batch_size, win_len, *other_dims))

        return return_batch
