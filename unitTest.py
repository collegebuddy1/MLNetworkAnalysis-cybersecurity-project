"""
Display main menu:
1. Train ( labeledData ) -> return parameters for the model ( features to use etc )
- GA randomly selects parameters and uses model on data
- evaluates and creates new parameters


2. Evaluate ( pcap data, model parameters ) -> return model results
- transform pcap to KDD format
- use model with trained parameters to evaluate

"""

import unittest
from Genetic import FeatureSet
from SVMClass import Dataset

class TestClass(unittest.TestCase):

    def test_init(self):
        featureSet = FeatureSet(20)
        featureSet.features = [0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,1,1,1,1,1]
        dataset = Dataset([20,20])
        removeIndices = dataset.adjustFeatures(featureSet)
        correctIndices = [0,1,2,5,6,10,11,12,13,14]
        self.assertTrue(removeIndices==correctIndices)

if __name__ == '__main__':
    unittest.main()
