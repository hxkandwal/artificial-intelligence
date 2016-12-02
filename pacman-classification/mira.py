# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        best_tuned_weights = {}
        best_classification_accuracy = None

        for c_value in Cgrid:
            local_weights = self.weights.copy()
            local_classification_accuracy = 0

            for iteration in range(self.max_iterations):
                local_iteration_classification_accuracy = 0

                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):

                    training_data_instance = trainingData[i]
                    training_data_label = trainingLabels[i]

                    # When we come to an instance (f,y)(f,y), we find the label with highest score
                    (max_label_score, best_label) = (None, None)
                    for label in self.legalLabels:
                        local_label_score = 0

                        for feature in self.features:
                            local_label_score += local_weights[label][feature] * training_data_instance[feature]

                        if max_label_score is None or max_label_score < local_label_score:
                            max_label_score = local_label_score
                            best_label = label

                    # adjust weights if arg-max label is different than training_data_label
                    if best_label != training_data_label:

                        # MIRA : choose	an update size that fixes the current mistake but, minimizes the change	to w
                        tuning_rate = min(c_value, ((local_weights[best_label] - local_weights[training_data_label]) *
                                                    training_data_instance + 1.0) / (2.0 * (training_data_instance *
                                                                                            training_data_instance)))

                        # update the weights with the tuning (learning) rate value.
                        for feature in self.features:
                            local_weights[best_label][feature] -= tuning_rate * training_data_instance[feature]
                            local_weights[training_data_label][feature] += tuning_rate * training_data_instance[feature]
                    else:
                        local_iteration_classification_accuracy += 1

                # update the local c-value classification accuracy.
                if local_iteration_classification_accuracy > local_classification_accuracy:
                    local_classification_accuracy = local_iteration_classification_accuracy

            if local_classification_accuracy > best_classification_accuracy:
                best_classification_accuracy = local_classification_accuracy
                best_tuned_weights = local_weights

        self.weights = best_tuned_weights

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


