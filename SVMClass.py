from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.utils import Bunch
import numpy as np

from sklearn.datasets import fetch_openml
from Genetic import *


class Classifier:

    def __init__(self, nuVal, gamVal):
        #look at own kernel & LSSVM
        self.model = svm.OneClassSVM(nu=nuVal, kernel='rbf', gamma=gamVal)#svm.SVC(kernel='rbf')
        self.performance = []
        results = []

    def train (self, dataset):
        
        trainData, testData, trainTarget, testTarget = train_test_split(dataset.samples, dataset.targets, test_size=0.3,random_state=108)
        self.model.fit( trainData)#, trainTarget)
        #print("testTarg:",testTarget)
        prediction = self.model.predict(testData)
        #print("prediction: " + str(prediction))

        self.calculatePerformance( testTarget, prediction)
        
        return

    def predictNew (self, unclassData):
        #results = ...
        return self.model.predict(unclassData)

    def calculatePerformance(self, test, prediction):
        self.performance = metrics.accuracy_score(test, prediction)
        #and so on other metrics
                                

    def getPerformance(self):
        return self.performance

    def updateModel(nuVal, gamVal):
        self.model = svm.OneClassSVM(nu=nuVal, kernel='rbf', gamma=gamVal)



class Main:

    def __init__(self):
        self.rawData = -1
        self.trainingData = -1
        self.testData = -1
        self.classifier = Classifier()


    def loadData( self, filename ):       
        #rawData = #... file reading code
        processedData = Dataset()
        processedData.clean(rawdata) # preprocessing
        self.trainingData, self.testData = processedData.split()
        
        return

    def trainModel(self, trainFile):
        #while performance is increasing:
        self.loadData(trainFile)
        # featureSet = FeatureSelector.ga(self.classifier)
        # vv this will be done in ga vv
        self.classifier.train(self.trainingData)
        self.classifier.getPerformance() #show metrics of trained clsf
        #update features and parameters based on GA
        return

    def classifyData(self, classFile ):
        #load and prep new File
        #self.classifier.predict()
        #self.classifier.showResults()
        return

    def chooseMenuOption(self,fn):

        functionDict = {
            "1" : self.trainModel(fn),
            "2" : self.classifyData(fn)
            }
        return functionDict.get(fn)

    def showMenu(self):

        # based on choice call other functions
        choice = ""
        filename = ""
        while choice != "q":
            choice = input("1. Train the model\n2. Classify new data")
            filename = input("Name of the file to use: ")
            self.chooseMenuOption(filename)
            
            
            
        return
                                
class Dataset:

    def __init__(self, shape): #data in form of one unorganized array not yet
        self.samples = np.empty((shape[0],shape[1]))
        self.targets = np.empty(shape[0])
        self.features = []
        

    def cleanData(self, raw):
        #sort data into 2 second and connection statistics
        #save stats into samples
        #save classification of each 2sec period into targets
        return

    def makeCopy(self):
        copy = Dataset(self.samples.shape)
        copy.samples = self.samples
        copy.targets = self.targets
        return copy

    # adapt this to take featureSet
    def removeFeatures(self, featureIndices):
        #cut columns out of samples
        self.samples = np.delete(self.samples, featureIndices, axis=1)
        return

    def adjustFeatures(self, featureSet):
        removeIndices =  []
        for i in range(len(featureSet.features)):
            if featureSet.features[i] == 0:
                removeIndices.append(i)
        self.samples = np.delete(self.samples, removeIndices, axis=1)
        return removeIndices

    def getData(self):
        return self.samples

    def getTargets(self):
        return self.targets
        
if __name__ == "__main__":
    
    #main = Main()
    #main.showMenu()

    cancer = fetch_openml(name='KDDCup99', version=1)
    #cancer = datasets.load_wine()
    print(cancer.details)
    
    print("Features: ", cancer.feature_names)
    print("Labels: ", np.unique(cancer.target))
    print("shape: ",cancer.data.shape)    


    dataset = Dataset(cancer.data.shape)
    dataset.samples = cancer.data
    dataset.targets = cancer.target

    #prepare for oneclass
    dataset.targets[ dataset.targets == "normal"] = 1
    dataset.targets[ dataset.targets != "normal"] = -1
    #print("Targetsss: ", dataset.targets)

    outliers = dataset.targets[dataset.targets == -1]
    outFraction = outliers.shape[0]/dataset.targets.shape[0]

    clf = Classifier(outFraction, 0.005)
    

    #dataset.removeFeatures([1,2,3,5,6,8])
    print("New shape: " + str(dataset.samples.shape))

    featureSelector = FeatureSelector()
    featureSelector.ga(2,clf,dataset)

    #clf.train(dataset)
    print("Performance: " + str(clf.getPerformance()))


        
        
    
