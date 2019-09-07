import os
import shutil

import pickle
import pytest

from .. import utils

# Test preprocess function 

@pytest.mark.parametrize('fruit, num_images, results', 
                         [
                             ('apples', 5, (4, 244, 244, 3)),
                             ('oranges', 5, (4, 244, 244, 3)),
                             ('pears', 5, (4, 244, 244, 3))
                         ]
                        )

def test_output_shape(fruit, num_images, results): 
    test_res = utils.preprocess(fruit, num_images).shape    
    assert test_res == results
    
@pytest.mark.parametrize('fruit, num_images', 
                         [
                             ('apples', 9999),
                             ('oranges', 9999),
                             ('pears', 9999)
                         ]
                        )

def test_large_num_images(fruit, num_images): 
    assert utils.preprocess(fruit, num_images) is None
    
# Test preprocess_all function

def test_data_folder_create():
    if os.path.isdir('{0}/data'.format(os.getcwd())):
        shutil.rmtree('{0}/data'.format(os.getcwd()))
        
    utils.preprocess_all(num_images=1, fruit_classes=['apples'])
    assert os.path.isdir('{0}/data'.format(os.getcwd()))

@pytest.fixture(scope='class')
def preprocess_all(num_images, fruit_classes):

    if os.path.isdir('{0}/data'.format(os.getcwd())):
            shutil.rmtree('{0}/data'.format(os.getcwd()))
    return utils.preprocess_all(num_images, fruit_classes)

@pytest.mark.parametrize('num_images, fruit_classes, num_classes', 
                         [
                             (5, ['apples'], 1),
                             (5, ['oranges'], 1),
                             (5, ['pears'], 1),
                             (5, ['apples', 'oranges'], 2),
                             (5, ['apples', 'oranges', 'pears'], 3)                             
                         ],
                         scope="class"
                        )

class TestPreprocessAll:
    
    def test_num_classes(self, preprocess_all, num_classes):
        assert len(os.listdir('{0}/data'.format(os.getcwd()))) == num_classes
    
    def test_pickle_save(self, preprocess_all, fruit_classes, num_classes):
        for fruit in fruit_classes:
            assert os.path.isfile('{0}/data/x_{1}.pkl'.format(os.getcwd(),fruit))