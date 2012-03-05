
"""
The coordinate-wise path setup.py script.
"""

from numpy.distutils.misc_util import Configuration

def configuration(parent_package='',top_path=None, package_name='cwpath'):

    config = Configuration('cwpath',parent_package,top_path)

    #config.add_extension('lasso',
    #                     sources = ["lasso.c"],
    #                     )
    config.add_extension('graphnet',
                         sources = ["graphnet.c"],
                         )
    config.add_extension('regression',
                         sources = ["regression.c"],
                         )
    config.add_extension('cwpath',
                         sources = ["cwpath.c"],
                         )
    
    
    return config

if __name__ == '__main__':

    from numpy.distutils.core import setup

    setup(**configuration(top_path='',
                          package_name='cwpath',
                          ).todict())
