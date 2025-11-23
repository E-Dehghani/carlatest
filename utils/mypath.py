
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        if database == 'msl' or database == 'smap':
            return '/home/edehghan/Data/CARLA-main/datasets/MSL_SMAP'
        elif database == 'ucr':
            return '/home/edehghan/Data/CARLA-main/datasets/UCR'
        elif database == 'yahoo':
            return '/home/edehghan/Data/CARLA-main/datasets/Yahoo'
        elif database == 'smd':
            return '/home/edehghan/Data/CARLA-main/datasets/SMD'
        elif database == 'swat':
            return '/home/edehghan/Data/CARLA-main/datasets/SWAT'
        elif database == 'wadi':
            return '/home/edehghan/Data/CARLA-main/datasets/WADI'
        elif database == 'kpi':
            return '/home/edehghan/Data/CARLA-main/datasets/KPI'
        elif database == 'swan':
            return '/home/edehghan/Data/CARLA-main/datasets/Swan'
        elif database == 'gecco':
            return '/home/edehghan/Data/CARLA-main/datasets/GECCO'
        
        else:
            raise NotImplementedError
