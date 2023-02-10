import unittest
import pandas as pd
from unittest.mock import MagicMock
import sys
sys.path.insert(0, '/home/cyjerox/Downloads/poc-to-prod-capstone/poc-to-prod-capstone/preprocessing/preprocessing')
import utils

class TestBaseTextCategorizationDataset(unittest.TestCase):
    def test__get_num_train_samples(self):
        """
        we want to test the class BaseTextCategorizationDataset
        we use a mock which will return a value for the not implemented methods
        then with this mocked value, we can test other methods
        """
        # we instantiate a BaseTextCategorizationDataset object with batch_size = 20 and train_ratio = 0.8
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_samples = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_samples(), 80)

    def test__get_num_train_batches(self):
        """
        same idea as what we did to test _get_num_train_samples
        """
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_train_batches = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_batches(), 8)

    def test__get_num_test_batches(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        # we mock _get_num_samples to return the value 100
        base._get_num_train_batches = MagicMock(return_value=100)
        # we assert that _get_num_train_samples will return 100 * train_ratio = 80
        self.assertEqual(base._get_num_train_batches(), 2)

    def test_get_index_to_label_map(self):
        base = utils.BaseTextCategorizationDataset()
        base._get_label_list = MagicMock(return_value=["php","python3","C#"])
        self.assertEqual(base.get_index_to_label_map(),{"1" : "php", "2" : "python" , "3" : "C#"})

    def test_index_to_label_and_label_to_index_are_identity(self):
        base = utils.BaseTextCategorizationDataset
        base._get_label_list = MagicMock(return_value=["php","python3","C#"])
        base.get_label_to_index_map()

    def test_to_indexes(self):
        base = utils.BaseTextCategorizationDataset(20, 0.8)
        base._get_label_list = MagicMock(return_value=['val1', 'val2', 'val3'])
        base.get_index_to_label_map()
        self.assertEqual(base.to_indexes(['val1', 'val3']), [0, 2])


class TestLocalTextCategorizationDataset(unittest.TestCase):
    def test_load_dataset_returns_expected_data(self):
        # we mock pandas read_csv to return a fixed dataframe
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_1', 'id_2'],
            'tag_name': ['tag_a', 'tag_b'],
            'tag_id': [1, 2],
            'tag_position': [0, 1],
            'title': ['title_1', 'title_2']
        }))
        # we instantiate a LocalTextCategorizationDataset (it'll use the mocked read_csv), and we load dataset
        dataset = utils.LocalTextCategorizationDataset.load_dataset("fake_path", 1)
        # we expect the data after loading to be like this
        expected = pd.DataFrame({
            'post_id': ['id_1'],
            'tag_name': ['tag_a'],
            'tag_id': [1],
            'tag_position': [0],
            'title': ['title_1']
        })
        # we confirm that the dataset and what we expected to be are the same thing
        pd.testing.assert_frame_equal(dataset, expected)

    def test__get_num_samples_is_correct(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_test_1', 'id_test_2'],
            'tag_name': ['tag_test_1', 'tag_test_1'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))
        test_dataset = utils.LocalTextCategorizationDataset("path", batch_size=1, train_ratio=0.5,
                                                       min_samples_per_label=1)
        self.assertEqual(test_dataset._get_num_samples(), 2)

    def test_get_train_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_test_1', 'id_test_2','id_test_3','id_test_4'],
            'tag_name': ['tag_test_1', 'tag_test_1', 'tag_test_2', 'tag_test_2'],
            'tag_id': [1, 1, 2,2],
            'tag_position': [0, 0, 0, 0],
            'title': ['title_1', 'title_2','title_3','title_4']
        }))
        dataset = utils.LocalTextCategorizationDataset("path", batch_size=2, train_ratio=0.5, min_samples_per_label=1)
        x_batch, y_batch = dataset.get_train_batch()
        self.assertEqual(len(x_batch), 2)
        self.assertEqual(y_batch.shape, (2, 2))

    def test_get_test_batch_returns_expected_shape(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_test_1', 'id_test_2', 'id_test_3', 'id_test_4', 'id_test_5', 'id_test_6'],
            'tag_name': ['tag_test_1', 'tag_test_1', 'tag_test_2', 'tag_test_2', 'tag_test_3', 'tag_test_3'],
            'tag_id': [1, 1, 2, 2, 3, 3],
            'tag_position': [0, 0, 0, 0, 0, 0],
            'title': ['title_1', 'title_2', 'title_3', 'title_4', 'title_5', 'title_6']
        }))

        dataset = utils.LocalTextCategorizationDataset("path", batch_size=1, train_ratio=0.5,
                                                       min_samples_per_label=1)
        x_batch, y_batch = dataset.get_test_batch()
        self.assertEqual(len(x_batch), 1)
        self.assertEqual(y_batch.shape, (1, 3))

    def test_get_train_batch_raises_assertion_error(self):
        pd.read_csv = MagicMock(return_value=pd.DataFrame({
            'post_id': ['id_test_1', 'id_test_2'],
            'tag_name': ['tag_test_1', 'tag_test_1'],
            'tag_id': [1, 2],
            'tag_position': [0, 0],
            'title': ['title_1', 'title_2']
        }))

        with self.assertRaises(AssertionError):
            utils.LocalTextCategorizationDataset("path", batch_size=3, train_ratio=0.5, min_samples_per_label=1)

