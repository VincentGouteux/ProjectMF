import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def loadData(directory):
    """
        Takes as input the directory of the dataset.
        Outputs two pandas frames: ratings and movies.
    """
    ratings = pd.read_csv(directory + '/ratings.csv',
                          usecols=['userId', 'movieId', 'rating'])
    movies = pd.read_csv(directory + 'movies.csv')
    return ratings, movies


def Df2Numpy(ratings):
    ratingsMatrix = ratings.pivot(index='userId',
                                  columns='movieId',
                                  values='rating')
    ratingsMatrix = ratingsMatrix.fillna(0)
    R = ratingsMatrix.to_numpy()
    return R


def dataAnalysis(R):
    nUsers = R.shape[0]
    nMovies = R.shape[1]
    
    # Number of rated movies per user
    ratingsPerUser = [sum(R[i, :] > 0) for i in range(nUsers)]
    maxRatingsPerUser = max(ratingsPerUser)
    minRatingsPerUser = min(ratingsPerUser)
    avgRatingsPerUser = sum(ratingsPerUser) / len(ratingsPerUser)
    print("Max ratings per user : ", maxRatingsPerUser)
    print("Min ratings per user : ", minRatingsPerUser)
    print("Avg ratings per user : ", avgRatingsPerUser)
    usersNoRatings = len([u for u in ratingsPerUser if u == 0])
    print("Users with no ratings : ", usersNoRatings)
    
    # Number of ratings per movie
    ratingsPerMovie = [sum(R[:, j] > 0) for j in range(nMovies)]
    maxRatingsPerMovie = max(ratingsPerMovie)
    minRatingsPerMovie = min(ratingsPerMovie)
    avgRatingsPerMovie = sum(ratingsPerMovie) / len(ratingsPerMovie)
    print("Max ratings per movie : ", maxRatingsPerMovie)
    print("Min ratings per movie : ", minRatingsPerMovie)
    print("Avg ratings per movie : ", avgRatingsPerMovie)
    moviesNoRatings = len([m for m in ratingsPerMovie if m == 0])            
    print("Movies with no ratings : ", moviesNoRatings)


class MatrixFactorization():
    """
    A simple Matrix Factorization Class
    Assumes ratings is a m x n Numpy array
    nFactors is the intermediate dimension k of the Matrices U and V
    lambdaReg and muReg are regularization parameters
    """
    def __init__(self,
                 ratings,
                 nFactors=10,
                 alpha=0.01,
                 lambdaReg=0.0,
                 muReg=0.0,
                 maxIter=50,
                 epsilon=0.1,
                 trainFrac=0.8,
                 valFrac=0.1,
                 testFrac=0.1):
        self.R = ratings
        self.nFactors = nFactors
        self.lambdaReg = lambdaReg
        self.alpha = alpha
        self.muReg = muReg
        self.maxIter = maxIter
        self.epsilon = epsilon
        self.nUsers, self.nMovies = ratings.shape
        self.trainFrac = trainFrac
        self.valFrac = valFrac
        self.testFrac = testFrac

        self.U = np.random.normal(scale=(1. / self.nFactors),
                                  size=(self.nUsers, self.nFactors))
        self.V = np.random.normal(scale=(1. / self.nFactors),
                                  size=(self.nMovies, self.nFactors))

    def matrix2Samples(self, matrix):
        """
        Convert matrix to a list of tuples (row, column, value) with only
        nonzero entries of the matrix
        """
        samples = [(i, j, matrix[i, j]) for i in range(matrix.shape[0])
                   for j in range(matrix.shape[1]) if matrix[i, j] > 0]
        return samples

    def samples2Matrix(self, samples, m, n):
        """
        Convert list of tuples (row, column, value) to a matrix
        of size m x n
        """
        matrix = np.zeros(m, n)
        for s in samples:
            i, j, v = s
            matrix[i, j] = v
            return matrix

    def randomInit(self):
        """
        Initialise target matrices U and V using normally distributed
        numbers
        """
        self.U = np.random.normal(scale=(1. / self.nFactors),
                                  size=(self.nUsers, self.nFactors))
        self.V = np.random.normal(scale=(1. / self.nFactors),
                                  size=(self.nMovies, self.nFactors))

        # def splitTrainValTest(self):
        #     """
        #     Shuffle the samples and sends back a partition for training,
        #     validation and testing
        #     """

    def splitTrainValSets(self, nGrades=10):
        """
        Split the observed data (nonzero entries) into a training set
        and a validation set by removing 10 grades per user and assigning them
        to the validation set
        """
        valMatrix = np.zeros(self.R.shape)
        trainMatrix = self.R.copy()
        for i in range(self.R.shape[0]):
            valRatingsIds = np.random.choice(self.R[i, :].nonzero()[0],
                                             size=nGrades,
                                             replace=False)
            trainMatrix[i, valRatingsIds] = 0
            valMatrix[i, valRatingsIds] = self.R[i, valRatingsIds]

        assert (np.all(trainMatrix * valMatrix) == 0)
        trainSamples = self.matrix2Samples(trainMatrix)
        valSamples = self.matrix2Samples(valMatrix)
        return trainSamples, valSamples

    def stochasticGradientDescentPass(self, trainSamples):
        for s in trainSamples:
            i, j, _ = s
            r_hat_ij = np.dot(self.U[i, :], self.V.T[:, j])
            eij = self.R[i, j] - r_hat_ij
            tmpU = np.zeros(self.nFactors)
            tmpV = np.zeros(self.nFactors)
            for q in range(self.nFactors):
                tmpU[q] = self.U[i, q] + self.alpha * (
                    eij * self.V[j, q] - self.lambdaReg * self.U[i, q])
                tmpV[q] = self.V[j, q] + self.alpha * (
                    eij * self.U[i, q] - self.muReg * self.V[j, q])

            self.U[i, :] = tmpU[:]
            self.V[j, :] = tmpV[:]

        return self.U, self.V

    def stochasticGradientDescent(self, logging=True):
        history = {'trainErrors': [], 'valErrors': []}
        self.randomInit()
        trainSamples, valSamples = self.splitTrainValTestSets()
        for i in range(self.maxIter):
            np.random.shuffle(trainSamples)
            U, V = self.stochasticGradientDescentPass(trainSamples)
            trainError = self.computeTotalErrorSamples(trainSamples)

            if logging:
                valError = self.computeTotalErrorSamples(valSamples)
                history['trainErrors'].append(trainError)
                history['valErrors'].append(valError)
            if trainError < self.epsilon:
                break

        history['trainError'] = self.computeTotalErrorSamples(trainSamples)
        history['valError'] = self.computeTotalErrorSamples(valSamples)

        return U, V, history

    def plotHistory(self, history):
        plt.plot(history['trainErrors'], label='Training Error')
        plt.plot(history['valErrors'], label='Validation Error')
        plt.title('Error(iteration)')
        plt.ylabel('Error')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

    def computeTotalErrorSamples(self, data):
        error = 0
        if len(data) == 0:
            return 0
        for s in data:
            i, j, r = s
            error += (r - np.dot(self.U[i, :], self.V.T[:, j]))**2
        return math.sqrt(error / len(data))

    def computeApproximatedMatrix(self, U, V):
        return np.dot(U, V.T)
