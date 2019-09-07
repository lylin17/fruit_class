import os
import shutil

import numpy as np
import pickle
import pytest

from .. import utils

# Test Data class 

def test_data_folder_create():
    if os.path.isdir('{0}/data'.format(os.getcwd())):
        shutil.rmtree('{0}/data'.format(os.getcwd()))
        
    data = utils.Data(fruit_classes=['apples'])
    data.get_data()
    assert os.path.isdir('{0}/data'.format(os.getcwd()))
    
@pytest.fixture(scope='class')
def data(fruit_classes):
    if os.path.isdir('{0}/data'.format(os.getcwd())):
        shutil.rmtree('{0}/data'.format(os.getcwd()))
    
    data = utils.Data(fruit_classes)
    data.get_data()
    data.unpickle()
       
    return data

@pytest.mark.parametrize('fruit_classes, num_classes, num_images, test_size, x_train_shape, x_val_shape, y_train_shape, y_val_shape', 
                         [
                             (['apples'], 1, 10, 0.3, (7, 244, 244 , 3), (3, 244, 244 , 3), (7, 1), (3, 1)),
                             (['apples', 'oranges'], 2, 10, 0.3, (14, 244, 244 , 3), (6, 244, 244 , 3), (14, 2), (6, 2)),
#                             (['oranges', 'pears'], 2, 10, 0.2, (16, 244, 244 , 3), (8, 244, 244 , 3), (16, 2), (8, 2)),
                             (['apples', 'oranges', 'pears'], 3, 10, 0.2, (24, 244, 244 , 3), (6, 244, 244 , 3), (24, 3), (6, 3))
                         ],
                         scope="class"
                        )
    
class TestData:    
    def test_download(self, data, fruit_classes, num_classes, num_images, test_size, x_train_shape, x_val_shape, y_train_shape, y_val_shape):
        for fruit in fruit_classes:
            assert os.path.isfile('{0}/data/x_{1}.pkl'.format(os.getcwd(),fruit))
            
    def test_num_classes(self, data, num_classes, num_images, test_size, x_train_shape, x_val_shape, y_train_shape, y_val_shape):
        assert len(os.listdir('{0}/data'.format(os.getcwd()))) == num_classes
        
    def test_unpickle_shape(self, data, fruit_classes, num_classes, num_images, test_size, x_train_shape, x_val_shape, y_train_shape, y_val_shape):
        mapping = {'apples': data.x_apple, 'oranges': data.x_orange, 'pears': data.x_pear}
        for fruit in fruit_classes:            
            assert mapping[fruit].shape[1:] == (244,244,3)
    
    def test_split_data_shape(self, data, num_classes, num_images, test_size, x_train_shape, x_val_shape, y_train_shape, y_val_shape):
        x_train, x_val, y_train, y_val = data.format_split(num_images, test_size)
        assert x_train.shape == x_train_shape
        assert y_train.shape == y_train_shape
        assert x_val.shape == x_val_shape
        assert y_val.shape == y_val_shape

# Test Model Class        
        
@pytest.fixture(scope='class')
def model_and_data():    
    model = utils.Model()
    
    #Toy data for x_train, y_train, x_val, y_val
    np.random.seed(42)
    x_train = np.random.rand(20, 244, 244, 3)
    x_val = np.random.rand(5, 244, 244, 3)
    y_train = np.eye(3)[np.random.choice([0,1,2],20)]
    y_val = np.eye(3)[np.random.choice([0,1,2],5)]    
    
    return model, x_train, x_val, y_train, y_val
    
class TestModel:
    def test_model_compile(self, model_and_data):
        model = model_and_data[0]
        x_train, x_val, y_train, y_val = model_and_data[1], model_and_data[2], model_and_data[3], model_and_data[4]
        
        evaluate = model.model.evaluate(x_val,y_val)            
        assert type(evaluate) is list 
        
    def test_model_output_after_fit(self, model_and_data):
        model = model_and_data[0]
        x_train, x_val, y_train, y_val = model_and_data[1], model_and_data[2], model_and_data[3], model_and_data[4]
        
        if os.path.isfile('{0}/model.h5'.format(os.getcwd())):
            os.remove('{0}/model.h5'.format(os.getcwd()))
        
        model.model_utils(rotation_range=10, horizontal_flip=True,
                          model_save='model.h5', monitor='val_loss', mode='min', period=1)
        model.fit(x_train, y_train, x_val, y_val, batch_size=10, epochs=1, verbose=0)
              
        assert os.path.isfile('{0}/model.h5'.format(os.getcwd()))
        
    def test_evaluate(self, model_and_data):
        model = model_and_data[0]
        x_train, x_val, y_train, y_val = model_and_data[1], model_and_data[2], model_and_data[3], model_and_data[4]
        
        evaluate = model.evaluate(x_val, y_val, model_save='model.h5')        
        assert type(evaluate) is list 
