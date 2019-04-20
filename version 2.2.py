import pandas as pd
import sys
import random
import pickle
import datetime
import time
import os.path
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


# DEFINED variables
LENGTH_OF_GESTURE = 20                              # length of the input table
LAYERS_SIZE = [1, 4]                                # min and max values (besides last layer)
NEURONS_PER_LAYER = [3, LENGTH_OF_GESTURE]          # min and max, which is length of gesture
ACTIVATION_FUNCTION = ['relu']                      # activation methods
LAST_LAYER = 'softmax'                              # this is always 'softmax'
MUTATION_RATE = 0.4                                 # chance that a variables gets changed

# DEFINED static variables
EPOCHS = 4000                                       # number of trainings for every n_splits
BATCH_SIZE = 64                                     # number of pictures used in a single go
N_SPLITS = 2                                        # number of redoing the training
GROUP_SIZE = 8                                      # number of groups that trains at the same time
TRAIN_LOOP = 100                                    # number of training / generations to train, negative = infinite
SHOW_PROCEDURAL_TRAIN = False                       # shows current training evolution (takes up print space)
OUTPUT_SAVE = "./saves/"                            # location of the output files

# global variables
log_text = "~~~ OUTPUT LOG ~~~\n\n"                 # set the title of the output log text


# variables needed for the training
class Variables:

    # initialize all variables
    def __init__(self):
        # starting variables / waiting to be overwritten
        self.length_of_gesture = LENGTH_OF_GESTURE

        self.layers_size = 2

        self.neurons_per_layer = []
        self.neurons_per_layer.append(LENGTH_OF_GESTURE)
        self.neurons_per_layer.append(LENGTH_OF_GESTURE/2)

        self.activation_function = []
        self.activation_function.append(ACTIVATION_FUNCTION[0])
        self.activation_function.append(ACTIVATION_FUNCTION[0])

    # clones the current Variable class
    def clone(self):
        clone = Variables()
        clone.layers_size = self.layers_size

        clone.neurons_per_layer = []
        for i in range(0, len(self.neurons_per_layer)):
            clone.neurons_per_layer.append(self.neurons_per_layer[i])

        clone.activation_function = []
        for i in range(0, len(self.activation_function)):
            clone.activation_function.append(self.activation_function[i])

        return clone


# this trains the AI
class KerasAI:

    # initialize important stuff
    def __init__(self):
        self.initialize()
        self.variables = Variables()    # initialize variables
        #self.run()

    # initializes important stuff
    def initialize(self):
        self.df = pd.read_csv('./training_data_1D/UGS_1D_dataset.csv')

        features = list(self.df.columns)
        features.remove('Class')

        self.x = self.df[features].values
        self.y = self.df['Class'].values
        # x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=1)

        self.input_dim = self.x.shape[1]
        self.output_dim = len(set(self.y))

    # constructs the layers for the AI v1.0 (old)
    def feedforward(self):
        model = Sequential()
        model.add(Dense(LENGTH_OF_GESTURE, input_dim=self.input_dim, kernel_initializer='normal', activation='relu'))
        model.add(Dense(NEURONS_PER_LAYER[0], kernel_initializer='normal', activation=LAST_LAYER))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # constructs the custom layers for the AI v1.2
    def feedforwardV1_2(self):
        model = Sequential()

        for i in range(0, len(self.variables.neurons_per_layer)):
            model.add(Dense(self.variables.neurons_per_layer[i], input_dim=self.input_dim, kernel_initializer='normal',
                            activation=self.variables.activation_function[i]))

        model.add(Dense(NEURONS_PER_LAYER[0], kernel_initializer='normal', activation=LAST_LAYER))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    # runs the AI
    def run(self):
        # save to log
        global log_text
        log_text = log_text + "WAITING...\n"

        print('WAITING...')
        # decides if the evolution of the training should be printed (shown) or not
        if SHOW_PROCEDURAL_TRAIN == False:
            vbs = 0
        else:
            vbs = 2

        # set values
        estimator = KerasClassifier(build_fn=self.feedforwardV1_2, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=vbs)
        kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=1)

        # encode class values as integers
        encoder = LabelEncoder()
        encoder.fit(self.y)
        encoded_y = encoder.transform(self.y)
        # convert integers to dummy variables (i.e. one hot encoded)
        dummy_y = np_utils.to_categorical(encoded_y)
        results = cross_val_score(estimator, self.x, dummy_y, cv=kfold)

        # save to log
        log_text = log_text + "Baseline: %.3f%% (%.2f%%)\nDONE.\n" % (results.mean()*100, results.std()*100)

        print("Baseline: %.3f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
        print('DONE.')

        return results.mean(), results.std(), estimator


# the way every entity of the AI works
class Memory:

    # initialize important stuff
    def __init__(self):
        self.ai = KerasAI()
        self.randomize()
        self.ai.variables = self.variables
        self.meanOutput = -1
        self.standardOutput = -1
        self.output = None

    # randomizes the variables, used only in the 1st generation
    def randomize(self):
        self.variables = Variables()
        self.variables.layers_size = random.randint(LAYERS_SIZE[0], LAYERS_SIZE[1])

        self.variables.neurons_per_layer = []
        self.variables.activation_function = []
        for i in range(0, self.variables.layers_size):
            if i == 0:
                self.variables.neurons_per_layer.append(LENGTH_OF_GESTURE)
            else:
                self.variables.neurons_per_layer.append(random.randint(NEURONS_PER_LAYER[0], self.variables.neurons_per_layer[i - 1]))
            self.variables.activation_function.append(random.choice(ACTIVATION_FUNCTION))

    # shows the variables
    def showVariables(self):
        # save to log
        global log_text
        log_text = log_text + "layers: " + str(self.variables.layers_size) + "\n"
        log_text = log_text + str(self.variables.neurons_per_layer) + " // neurons per layer\n"
        log_text = log_text + str(self.variables.activation_function) + " // activation functions\n"
        log_text = log_text + "Prediction: %.3f%%\n\n" % (self.meanOutput * 100)

        print("layers: " + str(self.variables.layers_size))
        print(str(self.variables.neurons_per_layer) + " // neurons per layer")
        print(str(self.variables.activation_function) + " // activation functions")
        print("Prediction: %.3f%%" % (self.meanOutput * 100))
        return ""

    # sets the variables (unused now)
    def setVariables(self, variables):
        self.variables = variables
        self.ai.variables = variables

    # sets the variables (unused for now)
    def setVariablesV2(self, variables):
        # COPY variables WITHOUT REFERENCE
        self.variables.layers_size = variables.layers_size
        self.variables.neurons_per_layer = []
        self.variables.activation_function = []

        for i in range(0, len(variables.neurons_per_layer) - 1):
            self.variables.neurons_per_layer.append(variables.neurons_per_layer[i])

        for i in range(0, len(variables.activation_function) - 1):
            self.variables.activation_function.append(variables.activation_function[i])

        # copy the second round of variables in ai
        self.ai.variables.layers_size = variables.layers_size
        self.ai.variables.neurons_per_layer = []
        self.ai.variables.activation_function = []

        for i in range(0, len(variables.neurons_per_layer) - 1):
            self.ai.variables.neurons_per_layer.append(variables.neurons_per_layer[i])

        for i in range(0, len(variables.activation_function) - 1):
            self.ai.variables.activation_function.append(variables.activation_function[i])

    # sets the variables, cloning the Variables class
    def cloneVariables(self, variables):
        self.variables = variables.clone()
        self.ai.variables = variables.clone()

    # clones the current memory
    def clone(self):
        clone = Memory()
        clone.cloneVariables(self.variables)
        return clone

    # mutates the characteristics of a memory with an x chance
    def mutate(self):
        mutation_rate = MUTATION_RATE   # chance that a variables gets changed
        change_maybe = random.random()  # a random number between 0 and 1

        # change the first value parameter
        if change_maybe < mutation_rate:

            # while the value is the same, keep randomizing again
            old_layer_size = self.variables.layers_size
            while self.variables.layers_size == old_layer_size:
                self.variables.layers_size = random.randint(LAYERS_SIZE[0], LAYERS_SIZE[1])

            # resize every array from variable class
            if self.variables.layers_size < old_layer_size:
                old_neurons_per_layer = self.variables.neurons_per_layer
                old_activation_function = self.variables.activation_function
                self.variables.neurons_per_layer = []
                self.variables.activation_function = []

                # if the layers shrank then reset the two arrays and keep the first n values
                for i in range(0, self.variables.layers_size):
                    self.variables.neurons_per_layer.append(old_neurons_per_layer[i])
                    self.variables.activation_function.append(old_activation_function[i])

                # set the first value in he array to 20 (LENGTH_OF_GESTURE)
                self.variables.neurons_per_layer[0] = LENGTH_OF_GESTURE

        # if the layers size got bigger then add additional values to the 2 arrays
        for i in range(0, self.variables.layers_size):
            if len(self.variables.neurons_per_layer) - 1 < i:
                self.variables.neurons_per_layer.append(random.randint(NEURONS_PER_LAYER[0], self.variables.neurons_per_layer[i - 1]))
                self.variables.activation_function.append(random.choice(ACTIVATION_FUNCTION))
            else:
                # check if neurons per layer should be changed
                change_maybe = random.random()
                if change_maybe < mutation_rate and len(self.variables.neurons_per_layer) > 1:
                    if i >= len(self.variables.neurons_per_layer) - 1:
                        self.variables.neurons_per_layer[i] = (random.randint(NEURONS_PER_LAYER[0], self.variables.neurons_per_layer[i - 1]))
                    # skip the first value in the array since it must always be 20 (LENGTH_OF_GESTURE)
                    elif i > 0:
                        self.variables.neurons_per_layer[i] = \
                            (random.randint(self.variables.neurons_per_layer[i + 1], self.variables.neurons_per_layer[i - 1]))

                # check if activation function should be changed
                change_maybe = random.random()
                if change_maybe < mutation_rate:
                    self.variables.activation_function[i] = (random.choice(ACTIVATION_FUNCTION))

        # this sets the variables to the AI too
        self.setVariables(self.variables)
        # reset outputs
        self.meanOutput = -1
        self.standardOutput = -1

    # run the KerasAI algorithm
    def run(self):
        self.meanOutput, self.standardOutput, self.output = self.ai.run()
        return self.meanOutput, self.standardOutput, self.output


# contains the memory
class NeuralAI:

    # initialize important stuff
    def __init__(self):
        self.memory = Memory()
        self.operationEnded = False
        self.efficiency = -1

    # show a specific type of printing output
    def show(self):
        print(self.memory.showVariables())

    # update the smart AI
    def update(self):
        self.memory.run()
        self.operationEnded = True

    # calculate AI's efficiency
    def calculateEfficiency(self):
        if self.operationEnded == True:
            self.efficiency = self.memory.meanOutput
        return self.efficiency

    # clone this Neural AI
    def clone(self):
        clone = NeuralAI()
        clone.memory = self.memory.clone()
        return clone


# behaviour of the group of neurons
class GroupTrain:

    # initialize important stuff
    def __init__(self):
        self.initializeNeurals()
        self.updateEnded = False
        self.bestNeural = None

    # initialize the number of neural AI you want (defined variable)
    def initializeNeurals(self):
        self.neural = [None] * GROUP_SIZE
        for i in range(GROUP_SIZE):
            self.neural[i] = NeuralAI()

    # print relevant information
    def show(self):
        # save to log
        global log_text
        log_text = log_text + "\n~~~~ ADITIONAL INFO ~~~~\n\n"

        print("\n~~~~ ADITIONAL INFO ~~~~\n")
        for i in range(GROUP_SIZE):
            # save to log
            log_text = log_text + "Memory " + str(i) + ":\n"
            log_text = log_text + "MEMORY LOCATION: " + str(hex(id(self.neural[i].memory.variables))) + "\n"

            print("Memory " + str(i) + ":")
            print("MEMORY LOCATION: " + str(hex(id(self.neural[i].memory.variables))))
            self.neural[i].show()

    # updated every neural
    def update(self):
        # save to log
        global log_text
        log_text = log_text + "\n~~~~ RUNNING ~~~~\n"

        print("\n~~~~ RUNNING ~~~~")

        if self.updateEnded == False:
            init = 0
        else:
            # save to log
            log_text = log_text + "\nMemory 0:\n(done last time)\nBaseline: %.3f%% \n" % (self.neural[0].efficiency * 100)

            print("\nMemory 0:\n(done last time)\nBaseline: %.3f%%" % (self.neural[0].efficiency * 100))
            init = 1
        for i in range(init, GROUP_SIZE):
            # save to log
            log_text = log_text + "\nMemory " + str(i) + ":\n"

            print("\nMemory " + str(i) + ":")
            self.neural[i].update()
        self.updateEnded = True

    # calculate every efficiency
    def calculateEfficiency(self):
        for i in range(GROUP_SIZE):
            self.neural[i].calculateEfficiency()

    # calculates efficiency for every neural memory
    def calculateEfficiencySum(self):
        self.efficiencySum = 0
        for i in range(GROUP_SIZE):
            self.efficiencySum += self.neural[i].calculateEfficiency()
        return self.efficiencySum

    # sets the next generation
    def naturalSelection(self):
        newNeural = [None] * GROUP_SIZE
        # / reference remains
        self.setBest()
        self.calculateEfficiencySum()

        # build the first best memory from the last generation / reference remains
        newNeural[0] = self.bestNeural

        # build the new neural memories
        for i in range(1, GROUP_SIZE):
            newNeural[i] = self.selectMemory().clone()

        # trying with .copy()
        #self.neural = newNeural.copy()
        for i in range(0, GROUP_SIZE):
            self.neural[i] = newNeural[i]

    # should randomly elect a memory as a parent, but regarding their probabilities
    def selectMemory(self):
        randomSelection = random.uniform(0, self.calculateEfficiencySum())
        goThroughMemory = 0
        # go through every memory
        for i in range(GROUP_SIZE):
            goThroughMemory += self.neural[i].calculateEfficiency()
            if goThroughMemory >= randomSelection:
                return self.neural[i]
        # should never get here
        return None

    # mutate all memories
    def mutateMemories(self):
        # save to log
        global log_text
        log_text = log_text + "MUTATING... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"

        print("MUTATING... ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        for i in range(1, GROUP_SIZE):
            # save to log
            log_text = log_text + "\n~~ Before:\nMemory " + str(i) + "\n"
            log_text = log_text + str(self.neural[i].memory.variables.layers_size) + "\n"
            log_text = log_text + str(self.neural[i].memory.variables.neurons_per_layer) + "\n"
            log_text = log_text + str(self.neural[i].memory.variables.activation_function) + "\n"

            print("\n~~ Before:")
            print("Memory " +str(i))
            print(self.neural[i].memory.variables.layers_size)
            print(self.neural[i].memory.variables.neurons_per_layer)
            print(self.neural[i].memory.variables.activation_function)

            log_text = log_text + "\nMEMORY LOCATION: " + str(hex(id(self.neural[i].memory.variables))) + "\n\n"
            print("\nMEMORY LOCATION: " + str(hex(id(self.neural[i].memory.variables))) + "\n")

            self.neural[i].memory.mutate()

            log_text = log_text + "~~ After: \nMemory " + str(i) + "\n"
            log_text = log_text + str(self.neural[i].memory.variables.layers_size) + "\n"
            log_text = log_text + str(self.neural[i].memory.variables.neurons_per_layer) + "\n"
            log_text = log_text + str(self.neural[i].memory.variables.activation_function) + "\n"

            print("~~ After: ")
            print("Memory " + str(i))
            print(self.neural[i].memory.variables.layers_size)
            print(self.neural[i].memory.variables.neurons_per_layer)
            print(self.neural[i].memory.variables.activation_function)

        log_text = log_text + "\nEND MUTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" + "\n"
        print("\nEND MUTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # set the best neural of the memory
    def setBest(self):
        maxNeural = 0
        for i in range(GROUP_SIZE):
            if self.neural[i].calculateEfficiency() > maxNeural:
                maxNeural = self.neural[i].calculateEfficiency()
                # save variables / reference remains
                self.bestNeural = self.neural[i]

    # save the best one to the specified folder
    def saveBestToFile(self, file):
        pickle.dump(self.bestNeural.memory.output, open(file, 'wb'))


# main class that runs the program
class MainClass:

    # initialize important stuff
    def __init__(self):
        # checks the input for errors
        self.checkForErrors()
        # start counting elapsed time
        self.startTime = time.time()

        # other stuff
        self.group = GroupTrain()
        self.condition = TRAIN_LOOP
        self.generation = 1

    # checks if the input location actually exits in order to avoid program errors
    def checkForErrors(self):
        if os.path.exists(OUTPUT_SAVE) == False:
            print("Please make sure that the OUTPUT_SAVE location exists.")
            print("Program closing ..\n")

            # closing
            tf.keras.backend.clear_session()
            exit(-1)

    # prints current generation
    def generationInfo(self):
        # save to log
        global log_text
        log_text = log_text + "\n~~~~ GENERATION " + str(self.generation) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" \
           "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"

        print("\n~~~~ GENERATION " + str(self.generation) + " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
                                              + "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        self.generation += 1

    # sets and prints current time of the program
    def showCurrentTime(self):
        now = datetime.datetime.now()
        print("-> " + str(now.day) + "d:" + str(now.month) + "m:" + str(now.year) + "y ")
        print("-> " + str(now.hour) + "h:" + str(now.minute) + "m:" + str(now.second) + "s \n")

        # save to log
        global log_text
        log_text = log_text + "-> " + str(now.day) + "d:" + str(now.month) + "m:" + str(now.year) + "y \n"
        log_text = log_text + "-> " + str(now.hour) + "h:" + str(now.minute) + "m:" + str(now.second) + "s \n\n"

        # prints every important input options
    def variablesInfo(self):
        # print current system version
        print("\n~~~~ SYSTEM VERSION ~~~~\n" + str(sys.version) + "\n")

        # save to log
        global log_text
        log_text = log_text + "\n~~~~ SYSTEM VERSION ~~~~\n" + str(sys.version) + "\n\n"

        # print starting time
        print("~~~~ STARTING TIME ~~~~")
        log_text = log_text + "~~~~ STARTING TIME ~~~~\n"

        self.showCurrentTime()

        # print options
        print("~~~~ INPUT OPTIONS ~~~~\n")
        print("- LENGTH_OF_GESTURE =        " + str(LENGTH_OF_GESTURE))
        print("- LAYERS_SIZE =              " + str(LAYERS_SIZE))
        print("- NEURONS_PER_LAYER =        " + str(NEURONS_PER_LAYER))
        print("- ACTIVATION_FUNCTION =      " + str(ACTIVATION_FUNCTION))
        print("- LAST_LAYER =               " + str(LAST_LAYER))
        print("- MUTATION_RATE =            " + str(MUTATION_RATE))
        print()
        print("- EPOCHS =                   " + str(EPOCHS))
        print("- BATCH_SIZE =               " + str(BATCH_SIZE))
        print("- N_SPLITS =                 " + str(N_SPLITS))
        print("- GROUP_SIZE =               " + str(GROUP_SIZE))
        print("- TRAIN_LOOP =               " + str(TRAIN_LOOP))
        print("- SHOW_PROCEDURAL_TRAIN =    " + str(SHOW_PROCEDURAL_TRAIN))
        print("- OUTPUT_SAVE =              " + "\"" + str(OUTPUT_SAVE) + "\"")

        # save to log
        log_text = log_text + "~~~~ INPUT OPTIONS ~~~~\n\n"
        log_text = log_text + "- LENGTH_OF_GESTURE =        " + str(LENGTH_OF_GESTURE) + "\n"
        log_text = log_text + "- LAYERS_SIZE =              " + str(LAYERS_SIZE) + "\n"
        log_text = log_text + "- NEURONS_PER_LAYER =        " + str(NEURONS_PER_LAYER) + "\n"
        log_text = log_text + "- ACTIVATION_FUNCTION =      " + str(ACTIVATION_FUNCTION) + "\n"
        log_text = log_text + "- LAST_LAYER =               " + str(LAST_LAYER) + "\n"
        log_text = log_text + "- MUTATION_RATE =            " + str(MUTATION_RATE) + "\n\n"
        log_text = log_text + "- EPOCHS =                   " + str(EPOCHS) + "\n"
        log_text = log_text + "- BATCH_SIZE =               " + str(BATCH_SIZE) + "\n"
        log_text = log_text + "- N_SPLITS =                 " + str(N_SPLITS) + "\n"
        log_text = log_text + "- GROUP_SIZE =               " + str(GROUP_SIZE) + "\n"
        log_text = log_text + "- TRAIN_LOOP =               " + str(TRAIN_LOOP) + "\n"
        log_text = log_text + "- SHOW_PROCEDURAL_TRAIN =    " + str(SHOW_PROCEDURAL_TRAIN) + "\n"
        log_text = log_text + "- OUTPUT_SAVE =              " + "\"" + str(OUTPUT_SAVE) + "\"\n"

        print("\n\n... STARTING TRAINING ...\n")
        # save to log
        log_text = log_text + "\n\n... STARTING TRAINING ...\n\n"

    # save the data log
    def saveLog(self):
        # declare the global variable so we can modify them
        global log_text

        # write to the output log (equivalent with what's been printed in the cmd
        with open(OUTPUT_SAVE + "Output_log.txt", "a") as myfile:
            myfile.write(log_text)

        # reset log_text
        log_text = ""

    # save the best / generation
    def saveData(self):
        # save the graph
        name = OUTPUT_SAVE + "Generation_" + str(self.generation - 1)
        self.group.saveBestToFile(name + str("_prediction_%.3f%%" % (self.group.bestNeural.efficiency * 100) + ".sav"))

        # save the data log
        self.saveLog()

        # save the notes related to that graph
        file = open(name + "_notes.txt", "w")
        file.write("~~~ Training Dataset ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")

        # get the specific training variables to print in the output txt
        file.write("Layers: " + str(self.group.neural[0].memory.variables.layers_size) + "\n")
        file.write(str(self.group.neural[0].memory.variables.neurons_per_layer) + "  (neurons per layer)\n")
        file.write(str(self.group.neural[0].memory.variables.activation_function) + "  (activation functions)\n\n")

        # print training accuracy
        file.write("Accuracy: " + str(self.group.bestNeural.memory.meanOutput) + "\n")
        file.write(str("Percentage: %.2f%%" % (self.group.bestNeural.efficiency * 100)) + "\n\n")

        # specify the specific graph so you can copy paste search it in the list in case you want to find it fast
        file.write("Name of the written model: " + "Generation_" + str(self.generation - 1) + str("_prediction_%.3f%%" % (self.group.bestNeural.efficiency * 100) + ".sav\n\n"))
        file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n")
        file.close()

    # run the training algorithm
    def run(self):
        # print initial variables set by you
        self.variablesInfo()

        # save to log
        global log_text
        # while the continuing condition it's still true, keep goings
        while(self.condition):
            self.condition -= 1

            # print and update current generation
            self.generationInfo()
            self.group.update()

            # save to log
            log_text = log_text + "\nBEFORE NATURAL SELECTION:\n"
            print("\nBEFORE NATURAL SELECTION:")

            self.group.show()

            # calculate efficiency for every neural and make natural selection
            self.group.calculateEfficiency()
            self.group.naturalSelection()

            # save to log
            log_text = log_text + "BEFORE MUTATION:\n"
            print("BEFORE MUTATION:")
            self.group.show()

            # save data before mutation, after that mutate the current child generation
            self.saveData()
            self.group.mutateMemories()

            # save to log
            log_text = log_text + "\nAFTER MUTATION:\n"
            print("\nAFTER MUTATION:")
            self.group.show()

        # save to log
        log_text = log_text + "~~~~ ENDING TIME ~~~~\n"
        # print ending time
        print("~~~~ ENDING TIME ~~~~")
        self.showCurrentTime()

        # save to log
        log_text = log_text + "-> Elapsed time: %ds <-\n\n" % (time.time() - self.startTime)
        log_text = log_text + "~~~~ END OF PROGRAM ~~~~\n\n\n\n"

        print("-> Elapsed time: %ds <-\n" % (time.time() - self.startTime))
        print("~~~~ END OF PROGRAM ~~~~\n\n")

        # write to the output log (equivalent with what's been printed in the cmd
        # save the data log
        self.saveLog()

        # closing
        tf.keras.backend.clear_session()


# run the program
y = MainClass()
y.run()
