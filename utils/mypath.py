
import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database=''):
        db_names = {'msl', 'smap', 'smd', 'power', 'yahoo', 'kpi', 'swat', 'wadi', 'gecco', 'swan', 'ucr'}
        assert(database in db_names)

        # Get current working directory and construct dataset paths
        base_dir = os.getcwd()
        datasets_dir = os.path.join(base_dir, 'datasets')

        if database == 'msl' or database == 'smap':
            return os.path.join(datasets_dir, 'MSL_SMAP')
        elif database == 'ucr':
            return os.path.join(datasets_dir, 'UCR')
        elif database == 'yahoo':
            return os.path.join(datasets_dir, 'Yahoo')
        elif database == 'smd':
            return os.path.join(datasets_dir, 'SMD')
        elif database == 'swat':
            return os.path.join(datasets_dir, 'SWAT')
        elif database == 'wadi':
            return os.path.join(datasets_dir, 'WADI')
        elif database == 'kpi':
            return os.path.join(datasets_dir, 'KPI')
        elif database == 'swan':
            return os.path.join(datasets_dir, 'Swan')
        elif database == 'gecco':
            return os.path.join(datasets_dir, 'GECCO')
        
        else:
            raise NotImplementedError
