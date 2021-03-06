"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32)):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """

    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    
    y = np.array([i[7:9] for i in images_files ])
    imgs = [np.array(cv2.imread(os.path.join(folder, f), 0)) for f in images_files]
    X = np.array([cv2.resize(x, tuple(size)).flatten() for x in imgs])

    return (X, y)



def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    M = X.shape[0]
    N = int(p*M)
    data = np.random.permutation(M)
    Xtrain = X[data[:N]]
    Xtest = X[data[N:]]
    ytrain = y[data[:N]]
    ytest = y[data[N:]]
    
    return (Xtrain, ytrain, Xtest, ytest)
    raise NotImplementedError


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    mean_face = np.mean(x, axis = 0)
    
    return mean_face
    raise NotImplementedError


def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    M = X.shape[1]
    mu = get_mean_face(X)
    sigma = np.dot((X-mu).T, (X-mu))
    u, v = np.linalg.eigh(sigma)
    
    return (v[::-1][:k].T, u[::-1][:k])
    raise NotImplementedError


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        self.weights = self.weights/sum(self.weights)
        for i in range(self.num_iterations):
            tree = WeakClassifier(self.Xtrain, self.ytrain, self.weights)
            
            tree.train()
            pred = tree.predict(self.Xtrain.T)
            mistakes = np.where(pred!=self.ytrain)
            eps = np.sum(self.weights[mistakes])
            alpha = 0.5*np.log((1-self.eps)/self.eps)
            
            self.weakClassifiers.append(tree)
            self.alphas.append(alpha)
            
            if eps>self.eps:
                self.weights = self.weights*np.exp(-alpha*pred*self.ytrain)

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        pred = self.predict(self.Xtrain)
        correct = np.where(pred == self.ytrain)[0].size
        incorrect = np.where(pred != self.ytrain)[0].size
        
        return (correct, incorrect)
        raise NotImplementedError

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        
        pred = np.array([tree.predict(X.T) for tree in self.weakClassifiers])
        predictions = pred.T*self.alphas 
        predictions = np.sum(predictions, axis = 1)
        predictions = np.sign(predictions)
        
        return predictions
        raise NotImplementedError


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)
        feat = np.ones(self.size)*255.
        h_ = int(self.size[0]/2)
        feat[h_:,:] = (126./255)*feat[h_:,:]
        p, q = self.position
        h, w = self.size
        img[p:p+h, q:q+w] = feat
        
        return img
        raise NotImplementedError

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)        
        feat = np.ones(self.size)*255.
        w_ = int(self.size[1]/2)
        feat[:,w_:] = (126./255)*feat[:,w_:]
        p, q = self.position
        h, w = self.size
        img[p:p+h, q:q+w] = feat
        
        return img
        raise NotImplementedError

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)
        feat = np.ones(self.size)*255.
        h_ = int(self.size[0]/3)
        feat[h_:2*h_,:] = (126./255)*feat[h_:2*h_,:]
        p, q = self.position
        h, w = self.size
        img[p:p+h, q:q+w] = feat
        
        return img
        raise NotImplementedError

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)
        feat = np.ones(self.size)*255.
        w_ = int(self.size[1]/3)
        feat[:,w_:2*w_] = (126./255)*feat[:,w_:2*w_]
        p, q = self.position
        h, w = self.size
        img[p:p+h, q:q+w] = feat
        
        return img
        raise NotImplementedError

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        img = np.zeros(shape)
        feat = np.ones(self.size)*255.
        h_ = int(self.size[0]/2)
        w_ = int(self.size[1]/2)
        feat[:h_,:w_] = (126./255)*feat[:h_,:w_]
        feat[h_:,w_:] = (126./255)*feat[h_:,w_:]
        p, q = self.position
        h, w = self.size
        img[p:p+h, q:q+w] = feat
        
        return img
        raise NotImplementedError

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        ii = ii.astype(np.int32)
        p, q = self.position
        p = p-1
        q = q-1
        if p<0:
            p=0
        if q<0:
            q=0
        h, w = self.size
        
        if self.feat_type == (2, 1):  # two_horizontal
            h_ = int(h/2)
            score = (ii[p,q] - ii[p+h_,q] - ii[p,q+w] + ii[p+h_,q+w]) - (ii[p+h_,q] - ii[p+h,q] - ii[p+h_,q+w] + ii[p+h,q+w])

        if self.feat_type == (1, 2):  # two_vertical
            w_ = int(w/2)
            score = (ii[p,q] - ii[p+h,q] - ii[p,q+w_] + ii[p+h,q+w_]) - (ii[p,q+w_] - ii[p,q+w] - ii[p+h,q+w_] + ii[p+h,q+w])

        if self.feat_type == (3, 1):  # three_horizontal
            h_ = int(h/3)
            score = (ii[p,q] - ii[p+h_,q] - ii[p,q+w] + ii[p+h_,q+w]) - (ii[p+h_,q] - ii[p+2*h_,q] - ii[p+h_,q+w] + ii[p+2*h_,q+w]) + (ii[p+2*h_,q] - ii[p+h,q] - ii[p+2*h_,q+w] + ii[p+h,q+w])
            
        if self.feat_type == (1, 3):  # three_vertical
            w_ = int(w/3)
            score = (ii[p,q] - ii[p+h,q] - ii[p,q+w_] + ii[p+h,q+w_]) - (ii[p,q+w_] - ii[p,q+2*w_] - ii[p+h,q+w_] + ii[p+h,q+2*w_]) + (ii[p,q+2*w_] - ii[p,q+w] - ii[p+h,q+2*w_] + ii[p+h,q+w])
            
        if self.feat_type == (2, 2):  # four_square
            h_ = int(h/2)
            w_ = int(w/2)
            score = - (ii[p,q] - ii[p+h_,q] - ii[p,q+w_] + ii[p+h_,q+w_]) + (ii[p+h_,q] - ii[p+h,q] - ii[p+h_,q+w_] + ii[p+h,q+w_]) + (ii[p,q+w_] - ii[p+h_,q+w_] - ii[p,q+w] + ii[p+h_,q+w]) - (ii[p+h_,q+w_] - ii[p+h_,q+w] - ii[p+h,q+w_] + ii[p+h,q+w])
            
        return score
        raise NotImplementedError


def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    
    integrals = images.copy()
    i = 0
    for image in images:
        image_ = np.cumsum(np.cumsum(image, axis=1), axis=0)
        integrals[i] = image_
        i=i+1
        
    return integrals
    raise NotImplementedError


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei-1, sizej-1]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        for i in range(num_classifiers):

            # TODO: Complete the Viola Jones algorithm
            weights = weights/sum(weights)
            scores = [hf.evaluate(ii) for ii in self.integralImages for hf in self.haarFeatures]
            n_feats = len(self.haarFeatures)
            n_iis = len(self.integralImages)
            X_ = np.reshape(scores, (n_iis, n_feats))
            y_ = self.labels
            vj = VJ_Classifier(X_, y_, weights)
            vj.train()
            preds = vj.predict(X_.T)
            eps = 1*(preds != self.labels)
            beta = vj.error/(1-vj.error)
            print(vj.error)
            weights = weights*(beta**(1-eps))
            alpha = np.log(1/beta)
            
            self.classifiers.append(vj)
            self.alphas.append(alpha)
            


    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        ii = convert_images_to_integral_images(images)

        scores = np.zeros((len(ii), len(self.haarFeatures)))
        
        for i, im in enumerate(ii):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]
        # Populate the score location for each classifier 'clf' in
        # self.classifiers.

        # Obtain the Haar feature id from clf.feature

        # Use this id to select the respective feature object from
        # self.haarFeatures

        # Add the score value to score[x, feature id] calling the feature's
        # evaluate function. 'x' is each image in 'ii'

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        threshold = 0.5*np.sum(self.alphas)
        for x in scores:
            # TODO
            pred = np.sum([alpha*h.predict(x) for alpha, h in zip(self.alphas, self.classifiers)])
            if pred>=threshold:
                H = 1
            else:
                H = -1
            result.append(H)

        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        out = image.copy()
        image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        face = [self.predict([image[i:i+24, j:j+24]]) for i in range(h-24) for j in range(w-24)]
        face = np.reshape(face, (h-24, w-24))
        ind_face = np.where(face==1)
        y, x = np.mean(ind_face, axis = 1).astype(int)
        
        out = cv2.rectangle(out, (x, y), (x+24, y+24), color=(0, 255, 0), thickness=1)
                
        cv2.imwrite(os.path.join('output', filename+'.png'), out)
        
        return None
        raise NotImplementedError
