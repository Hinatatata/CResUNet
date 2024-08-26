import os
import pandas as pd
from transformer.weather_transform import WeatherTransformer


class DataCreator:

    def __init__(self, weather_raw_dir, start_date, end_date, spatial_range, target_dim, downsample_mode,
                 area_name, train_period, val_period, test_period,
                 weather_freq=3, features=None, atm_dim=0, check_files=False,
                 rebuild=True, smooth=True, smooth_win_len=31):

        # Data paths
        self.data_dir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.data_dir, 'data')
        self.weather_raw_dir = weather_raw_dir

        # Parameters
        self.start_date = start_date
        self.end_date = end_date
        self.spatial_range = spatial_range
        self.weather_freq = weather_freq
        self.rebuild = rebuild

        # weather transformer parameters
        self.check_files = check_files
        self.features = features
        self.atm_dim = atm_dim
        self.target_dim = target_dim
        self.smooth = smooth
        self.smooth_win_len = smooth_win_len
        self.downsample_mode = downsample_mode

    def create_data(self):
        weather_folder = os.path.join(self.data_dir, 'data_dump')

        if not self.rebuild:
            print('Loading from saved path')
            path_list = self.__get_file_paths(weather_folder)
            if len(path_list) == 0:
                raise ValueError('{} folder is empty'.format(weather_folder))
            path_arr = self.__sort_files_by_date(paths=path_list)

        else:

            if not os.path.isdir(weather_folder):
                os.makedirs(weather_folder)
            print('Creating the weather data')

            weather_transformer = WeatherTransformer(file_dir=self.weather_raw_dir,
                                                     features=self.features,
                                                     atm_dim=self.atm_dim,
                                                     check=self.check_files,
                                                     freq=self.weather_freq,
                                                     target_dim=self.target_dim,
                                                     smooth=self.smooth,
                                                     smooth_win_len=self.smooth_win_len,
                                                     downsample_mode=self.downsample_mode)

            date_r = pd.date_range(start=self.start_date,
                                   end=self.end_date,
                                   freq=str(self.weather_freq) + 'D')

            weather_transformer.transform_range(date_range=date_r,
                                                spatial_range=self.spatial_range,
                                                save_dir=weather_folder)

            path_list = self.__get_file_paths(weather_folder)
            path_arr = self.__sort_files_by_date(paths=path_list)

        return path_arr

    @staticmethod
    def __get_file_paths(in_dir):

        file_list = []
        for file_name in os.listdir(in_dir):
            file_path = os.path.join(in_dir, file_name)
            file_list.append(file_path)

        return file_list

    def __sort_files_by_date(self, paths):

        dates = [os.path.basename(file).split('.')[0] for file in paths]
        date_df = pd.DataFrame(list(zip(paths, dates)),
                               columns=['paths', 'dates'])
        date_df['dates'] = pd.to_datetime(
            date_df['dates'], format="%Y-%m-%d_%H")
        date_df.sort_values(by='dates', inplace=True)

        indices = (self.start_date <= date_df['dates']) & \
                  (date_df['dates'] <= self.end_date)
        date_df = date_df[indices]
        out_path = date_df['paths'].values

        return out_path
