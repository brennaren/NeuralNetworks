import java.util.Properties;
import java.util.Scanner;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Network class that implements an N-layer feed forward neural network with an input activation layer,
 * hidden layers, and an output layer (total of N activation layers). It includes methods to set configurations, 
 * allocate memory, populate the network with weights, train or run the network, and output the results. The 
 * network will be trained using gradient descent and with back propagation.
 * 
 * Methods include:
 * public void initializeVariables()
 * public void initializeDerivedValues()
 * public void setManualConfigs()
 * public void loadConfigsFromFile()
 * public void loadWeightsFromFile()
 * public void printNetworkConfigs()
 * public void printTrainingParameters()
 * public void allocateNetworkMemory()
 * public void populateNetwork()
 * public void fillRandomWeights()
 * public double getRandomValue(double low, double high)
 * public void fillFileTestCases()
 * public void trainAll()
 * public void updateWeights(int caseIndex)
 * public double derivActivationFunction(double theta)
 * public double activationFunction(double theta)
 * public double sigmoid(double x)
 * public double derivSigmoid(double x)
 * public double tanh(double x)
 * public double derivTanh(double x)
 * public void runForTrainByCase(int caseIndex)
 * public void printTrainResults()
 * public void runAll()
 * public void runByCase(int caseIndex)
 * public void setUpTestCase(int caseIndex)
 * public void printRunResults()
 * public void printNetworkWeights()
 * public void printInputTable()
 * public void printTruthTableWithOutputs()
 * public void printInputOutputOnly()
 * public void startTimer()
 * public void endTimer()
 * 
 * @author Brenna Ren
 * @version November 21, 2025
 * Date of creation: September 9, 2025
 */
public class Network 
{
   public final static String DEFAULT_CONFIG_FILE_PATH = "defaultConfigs.properties";  // default config file path

   public final static int INPUT_LAYER_INDEX = 0;     // input layer index (0-indexed)
   public final static int FIRST_H_LAYER_INDEX = 1;   // first hidden layer index (0-indexed)

   public int numActivationLayers;     // number of activation layers
   public int[] numActivations;        // number of activations in each layer
   public String networkConfigString;  // string representation of the network configuration
   
   public int numConnectivityLayers;   // number of connectivity layers
   public int lastHLayerIndex;         // index of the last hidden layer
   public int outputLayerIndex;        // index of the output layer

   public double randomWeightMin;   // minimum random weight value
   public double randomWeightMax;   // maximum random weight value
   public int maxIterations;        // maximum number of training iterations before stopping
   public double errorThreshold;    // training stops when average error is below this threshold
   public double lambdaValue;       // learning factor used to control the magnitude of weight updates

   public boolean printNetworkSpecifics;  // whether to print network specifics after training/running
   public boolean printInputTable;        // whether to print the input table after training/running
   public boolean printTruthTable;        // whether to print the truth table after training/running
   public boolean printHiddenActivations; // whether to print the hidden activations after each run

   public double keepAlive;            // number of iterations between messages (or no output if it is set to zero)

   public String weightConfig;         // whether to use manually specified weights or random weights
   public boolean isTraining;          // whether the network is in training mode (true) or running mode (false)
   public boolean runAfterTraining;    // whether to run the network after training
   public boolean saveWeightsToFile;   // whether to save weights to a file after training
   
   private double[][] a;            // input activations
   private double[][][] weights;    // weights for connections between all layers [layer][from][to]
   private double[][] thetas;       // theta values that are calculated while finding the weight deltas
   private double[][] psis;         // psi values that are calculated while finding the weight deltas

   public String configFilePath;       // file path to load configurations from
   public String loadWeightsFilePath;  // file path to load weights from (binary path)
   public String saveWeightsFilePath;  // file path to save weights to (binary path)
   public String inputsFilePath;       //  file path to load test cases from
   public String outputsFilePath;      // file path to load expected outputs from
   
   private double averageError;  // average error across all test cases
   private int iteration;        // current training iteration

   private int numTestCases;           // number of test cases
   private double[][] testCaseInput;   // input values for all test cases
   private double[][] testCaseOutput;  // expected output values for all test cases

   private double startTime;  // for timing training/running
   private double endTime;    // for timing training/running

/**
 * Initializes the network's variables to default values.
 */
   public void initializeVariables()
   {
      averageError = Double.MAX_VALUE;
      iteration = 0;
   }

/**
 * Initializes the derived values of the network based on the user configured variables.
 */
   public void initializeDerivedValues()
   {
      numConnectivityLayers = numActivationLayers - 1;
      lastHLayerIndex = numActivationLayers - 2;
      outputLayerIndex = numActivationLayers - 1;
   }

/**
 * Initializes the configurations of the Network with default, hard-coded values.
 * These values can be changed by modifying this method.
 */
   public void setManualConfigs()
   {
      this.numActivationLayers = 4;
      this.networkConfigString = "2-2-1-3";

      this.randomWeightMin = 0.1;
      this.randomWeightMax = 1.5;
      this.maxIterations = 100000;
      this.errorThreshold = 0.0002;
      this.lambdaValue = 0.3;
      
      this.printNetworkSpecifics = false;
      this.printInputTable = true;
      this.printTruthTable = true;
      this.printHiddenActivations = false;
      
      this.weightConfig = "Random"; // "Manual" or "Load" or "Random"
      this.loadWeightsFilePath = "AND_OR_XOR/saved_AND_OR_XOR_weights.bin";
      this.saveWeightsFilePath = "AND_OR_XOR/saved_AND_OR_XOR_weights.bin";
      this.saveWeightsToFile = false;

      this.isTraining = true;
      this.runAfterTraining = true;

      this.numTestCases = 4;
      this.inputsFilePath = "AND_OR_XOR/AND_OR_XOR_inputs.txt";
      this.outputsFilePath = "AND_OR_XOR/AND_OR_XOR_outputs.txt";
   } // public void setManualConfigs()

/**
 * Loads the network configurations from a specified .properties file path.
 * If the file cannot be read, an exception is thrown.
 * The expected format is .properties with keys matching the configuration variable names.
 */
   public void loadConfigsFromFile()
   {
      try
      {
         Properties props = new Properties();
         InputStream input = new FileInputStream(configFilePath);
         props.load(input);

         this.numActivationLayers = Integer.parseInt(props.getProperty("numActivationLayers"));
         this.networkConfigString = props.getProperty("networkConfig");

         this.randomWeightMin = Double.parseDouble(props.getProperty("randomWeightMin"));
         this.randomWeightMax = Double.parseDouble(props.getProperty("randomWeightMax"));
         this.maxIterations = Integer.parseInt(props.getProperty("maxIterations"));
         this.errorThreshold = Double.parseDouble(props.getProperty("errorThreshold"));
         this.lambdaValue = Double.parseDouble(props.getProperty("lambdaValue"));

         this.printNetworkSpecifics = Boolean.parseBoolean(props.getProperty("printNetworkSpecifics"));
         this.printInputTable = Boolean.parseBoolean(props.getProperty("printInputTable"));
         this.printTruthTable = Boolean.parseBoolean(props.getProperty("printTruthTable"));
         this.printHiddenActivations = Boolean.parseBoolean(props.getProperty("printHiddenActivations"));

         this.keepAlive = Double.parseDouble(props.getProperty("keepAlive"));

         this.weightConfig = props.getProperty("weightConfig");
         this.loadWeightsFilePath = props.getProperty("loadWeightsFilePath");
         this.saveWeightsFilePath = props.getProperty("saveWeightsFilePath");
         this.saveWeightsToFile = Boolean.parseBoolean(props.getProperty("saveWeightsToFile"));

         this.isTraining = Boolean.parseBoolean(props.getProperty("isTraining"));
         this.runAfterTraining = Boolean.parseBoolean(props.getProperty("runAfterTraining"));

         this.numTestCases = Integer.parseInt(props.getProperty("numTestCases"));
         this.inputsFilePath = props.getProperty("inputsFilePath");
         this.outputsFilePath = props.getProperty("outputsFilePath");
      } // try
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + configFilePath + " — " + e.getMessage(), e);
      }
   } // public void loadConfigsFromFile()

/**
 * Loads weights from a specified binary file path into the network's weight arrays.
 * If the file cannot be read, an exception is thrown.
 * The expected format is the network configuration, followed by the weights 
 * in the appropriate order as doubles.
 * The indexing in the loops that iterate through the weights define the current layer
 * as the one to the left (so weights[n][k][j] would be between n and n+1).
 */
   public void loadWeightsFromFile()
   {
      try 
      {
         InputStream inputStream = new FileInputStream(loadWeightsFilePath);
         DataInput dataInputStream = new DataInputStream(inputStream);

         String fileNetworkConfig = dataInputStream.readUTF();

         if (!fileNetworkConfig.equals(networkConfigString))
         {
            inputStream.close();
            throw new IllegalArgumentException("Error: Weight configuration in file does not match network configuration.");
         }

         for (int n = 0; n < numConnectivityLayers; n++)
         {
            for (int k = 0; k < numActivations[n]; k++)
            {
               for (int j = 0; j < numActivations[n+1]; j++)
               {
                  weights[n][k][j] = dataInputStream.readDouble();
               }
            }
         }

         inputStream.close();
      } // try
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + loadWeightsFilePath);
      }
   } // public void loadWeightsFromFile()

/**
 * Echos the network configurations by printing them to the console.
 */
   public void printNetworkConfigs()
   {
      System.out.println("\n---------NETWORK CONFIGURATIONS---------");
      System.out.println("Configurations File Path: " + configFilePath);
      System.out.println("Test Cases Input File Path: " + inputsFilePath);
      System.out.println("Test Cases Output File Path: " + outputsFilePath);
      System.out.println("Network Config: " + networkConfigString);
      System.out.println("Print Network Specifics: " + printNetworkSpecifics);
      System.out.println("Print Input Table: " + printInputTable);
      System.out.println("Print Truth Table: " + printTruthTable);
      System.out.println("Print Hidden Activations: " + printHiddenActivations);
      System.out.println("Keep Alive Iterations: " + keepAlive);
      System.out.println("Weight Configuration: " + weightConfig);
      System.out.println("Mode: " + (isTraining ? "Training" : "Running"));
      System.out.println("Run After Training: " + runAfterTraining);
      System.out.println("Number of Test Cases: " + numTestCases);
   } // public void printNetworkConfigs()
   
/**
 * Echos the training parameters by printing them to the console.
 */
   public void printTrainingParameters()
   {
      System.out.println("\n---------TRAINING PARAMETERS---------");
      System.out.println("Random Weight Range: " + randomWeightMin + " to " + randomWeightMax);
      System.out.println("Max Iterations: " + maxIterations);
      System.out.println("Error Threshold: " + errorThreshold);
      System.out.println("Lambda Value: " + lambdaValue);
   }

/**
 * Allocates memory for the network's arrays based on the number of activations.
 * The indexing in the loops that iterate through the weights define the current layer
 * as the one to the left (so weights[n][k][j] would be between n and n+1).
 */
   public void allocateNetworkMemory()
   {
      numActivations = new int[numActivationLayers];
      for (int n = 0; n < numActivationLayers; n++)
      {
         String[] parsedNetworkConfig = networkConfigString.split("-");
         numActivations[n] = Integer.parseInt(parsedNetworkConfig[n]);
      }

      a = new double[numActivationLayers][];
      for (int n = 0; n < numActivationLayers; n++)
      {
         a[n] = new double[numActivations[n]];
      }

      weights = new double[numConnectivityLayers][][];
      for (int n = 0; n < numConnectivityLayers; n++)
      {
         weights[n] = new double[numActivations[n]][numActivations[n+1]];
      }

      testCaseInput = new double[numTestCases][numActivations[INPUT_LAYER_INDEX]];
      
      if (isTraining || printTruthTable)
      {
         testCaseOutput = new double[numTestCases][numActivations[outputLayerIndex]];
      }

      if (isTraining)
      {
         thetas = new double[numActivationLayers-1][];
         for (int n = FIRST_H_LAYER_INDEX; n <= lastHLayerIndex; n++)
         {
            thetas[n] = new double[numActivations[n]];
         }

         psis = new double[numActivationLayers][];
         for (int n = FIRST_H_LAYER_INDEX; n < numActivationLayers; n++)
         {
            psis[n] = new double[numActivations[n]];
         }
      } // if (isTraining)
   } // public void allocateNetworkMemory()

/**
 * Populates the network's weights either with manually specified weights from a file
 * or with random weights within the specified range.
 */
   public void populateNetwork()
   {
      if (weightConfig.equals("Load"))
      {
         loadWeightsFromFile();
      }
      else
      {
         fillRandomWeights();
      }

      fillFileTestCases();
   } // public void populateNetwork()
   
/**
 * Fills the weights array with randomized weights between randomWeightMin and randomWeightMax.
 */
   public void fillRandomWeights()
   {
      for (int n = 0; n < numConnectivityLayers; n++)
      {
         for (int k = 0; k < numActivations[n]; k++)
         {
            for (int j = 0; j < numActivations[n+1]; j++)
            {
               weights[n][k][j] = getRandomValue(randomWeightMin, randomWeightMax);
            }
         }
      }
   } // public void fillRandomWeights()

/**
 * Generates a random double value between the specified low and high values.
 * @param low the minimum value (inclusive)
 * @param high the maximum value (exclusive)
 * @return a random double between low and high
 */
   public double getRandomValue(double low, double high)
   {
      return Math.random() * (high - low) + low;
   }

/**
 * Loads the test cases for the network from a file (specified in configs).
 * The file should contain the input values and expected output values for each test case.
 * If the file cannot be read, an exception is thrown.
 * The expected format is one test case per line, with input values followed by output values,
 * all separated by spaces.
 */
   public void fillFileTestCases()
   {
      Scanner inputFileScanner;
      try
      {
         inputFileScanner = new Scanner(new File(inputsFilePath));
      }
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + inputsFilePath);
      }

      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
         {
            if (inputFileScanner.hasNextDouble())
            {
               testCaseInput[caseIndex][m] = inputFileScanner.nextDouble();
            }
            else
            {
               inputFileScanner.close();
               throw new IllegalArgumentException("Error: Not enough input values in test cases file.");
            }
         } // for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
      } // for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)

      inputFileScanner.close();
      Scanner outputFileScanner;

      if (isTraining || printTruthTable)
      {
         try
         {
            outputFileScanner = new Scanner(new File(outputsFilePath));
         }
         catch (Exception e)
         {
            throw new IllegalArgumentException("Error: Unable to open file at " + outputsFilePath);
         }

         for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
         {
            for (int i = 0; i < numActivations[outputLayerIndex]; i++)
            {
               if (outputFileScanner.hasNextDouble())
               {
                  testCaseOutput[caseIndex][i] = outputFileScanner.nextDouble();
               } // if (fileScanner.hasNextDouble())
               else
               {
                  outputFileScanner.close();
                  throw new IllegalArgumentException("Error: Not enough output values in test cases file.");
               }
            } // for (int i = 0; i < numActivations[outputLayerIndex]; i++)
         } // for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)

         outputFileScanner.close();
      } // if (isTraining || printTruthTable)
   } // public void fillFileTestCases()

/**
 * Trains the network using all training data until the average error is below the error threshold
 * or the maximum number of iterations is reached.
 */
   public void trainAll()
   {
      while (averageError > errorThreshold && iteration < maxIterations)
      {
         double totalError = 0.0;

         for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
         {
            setUpTestCase(caseIndex);
            totalError += runForTrainByCase(caseIndex);
            updateWeights(caseIndex);
         }

         totalError /= 2.0;
         iteration++;
         averageError = totalError / numTestCases;

         if ((keepAlive != 0) && ((iteration % keepAlive) == 0)) 
         {
            System.out.printf("Iteration %d, Error = %f\n", iteration, averageError);
         }
      } // while (averageError > errorThreshold && iteration < maxIterations)
   } // public void trainAll()


/**
 * Updates the weights for the network based on the calculated psis from the outputs and the activations.
 * Uses the calculations outlined in the design document. Stores only the psis and weights.
 * @param caseIndex the index of the test case to use
 */
   public void updateWeights(int caseIndex)
   {
      for (int n = lastHLayerIndex; n > FIRST_H_LAYER_INDEX; n--)
      {
         for (int k = 0; k < numActivations[n]; k++)
         {
            double omega = 0.0;

            for (int j = 0; j < numActivations[n+1]; j++)
            {
               omega += psis[n+1][j] * weights[n][k][j];
               weights[n][k][j] += lambdaValue * a[n][k] * psis[n+1][j];
            }
         
            psis[n][k] = omega * derivActivationFunction(thetas[n][k]);
         } // for (int j = 0; j < numActivations[n]; j++)
      } // for (int n = lastHLayerIndex; n > FIRST_H_LAYER_INDEX; n--)
      
      int n = FIRST_H_LAYER_INDEX;
      for (int k = 0; k < numActivations[n]; k++)
      {
         double omega = 0.0;

         for (int j = 0; j < numActivations[n+1]; j++)
         {
            omega += psis[n+1][j] * weights[n][k][j];
            weights[n][k][j] += lambdaValue * a[n][k] * psis[n+1][j];
         }

         psis[n][k] = omega * derivActivationFunction(thetas[n][k]);
         
         for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
         {
            weights[INPUT_LAYER_INDEX][m][k] += lambdaValue * a[INPUT_LAYER_INDEX][m] * psis[n][k];
         }
      } // for (int k = 0; k < numActivations[n]; k++)
   } // public void updateWeights(int caseIndex)

/**
 * Calculates the derivative of the activation function.
 * This can be modified to implement different activation functions as needed.
 * @param theta the input value to the derivative of the activation function
 * @return the output of the derivative of the activation function
 */
   public double derivActivationFunction(double theta)
   {
      return derivSigmoid(theta);
   }

/**
 * Applies the activation function to the given theta value.
 * This can be modified to implement different activation functions as needed.
 * @param theta the input value to the activation function
 * @return the output of the activation function
 */
   public double activationFunction(double theta)
   {
      return sigmoid(theta);
   }

/**
 * Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 * @param x the input value to the sigmoid function
 * @return the output of the sigmoid function
 */
   public double sigmoid(double x)
   {
      return 1.0 / (1.0 + Math.exp(-x));
   }

/**
 * Derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x))
 * @param x the input value to the derivative of the sigmoid function
 * @return the output of the derivative of the sigmoid function
 */
   public double derivSigmoid(double x)
   {
      double sigmoidValue = sigmoid(x);
      return sigmoidValue * (1.0 - sigmoidValue);
   }

/**
 * Hyperbolic tangent activation function tanh(x) = (e^x - e^-x) / (e^x + e^-x).
 * Modified to prevent NAN errors for large values of abs(x). See design document for details.
 * @param x the input value to the tanh function
 * @return the output of the tanh function
 */
   public double tanh(double x)
   {
      double epsilon = (x > 0) ? 1.0 : -1.0;
      double epsilonExp = Math.exp(epsilon * 2.0 * x);
      return epsilon * ((epsilonExp - 1.0) / (epsilonExp + 1.0));
   }

/**
 * Derivative of the tanh function: f'(x) = 1 - (f(x))^2 where f(x) is the tanh function.
 * @param x the input value to the derivative of the tanh function
 * @return the output of the derivative of the tanh function
 */
   public double derivTanh(double x)
   {
      double tanhValue = tanh(x);
      return 1.0 - (tanhValue * tanhValue);
   }

/**
 * Runs the network for training for a specific test case index, calculating the hidden activations and output.
 * This saves the theta values, as they are needed during training.
 * @param caseIndex  the index of the test case to run
 */
   public double runForTrainByCase(int caseIndex)
   {
      double error = 0.0;

      for (int n = FIRST_H_LAYER_INDEX; n <= lastHLayerIndex; n++)
      {
         for (int j = 0; j < numActivations[n]; j++)
         {
            thetas[n][j] = 0.0;

            for (int k = 0; k < numActivations[n-1]; k++)
            {
               thetas[n][j] += a[n-1][k] * weights[n-1][k][j];
            }

            a[n][j] = activationFunction(thetas[n][j]);
         } // for (int j = 0; j < numActivations[n]; j++)
      } // for (int n = FIRST_H_LAYER_INDEX; n <= lastHLayerIndex; n++)

      int n = outputLayerIndex;
      for (int i = 0; i < numActivations[n]; i++)
      {
         double F_theta = 0.0;

         for (int j = 0; j < numActivations[n-1]; j++)
         {
            F_theta += a[n-1][j] * weights[n-1][j][i];
         }

         a[n][i] = activationFunction(F_theta);
         double F_omega = testCaseOutput[caseIndex][i] - a[n][i];
         psis[n][i] = F_omega * derivActivationFunction(F_theta);
         error += F_omega * F_omega;
      } // for (int i = 0; i < numActivations[n]; i++)

      return error;
   } // public void runForTrainByCase(int caseIndex)

/**
 * Prints the training results, including the number of iterations, final average error,
 * and optionally the network specifics.
 */
   public void printTrainResults()
   {
      System.out.println("\n---------TRAINING RESULTS---------");
      System.out.println("Iterations: " + iteration);
      System.out.printf("Final Average Error: %.6f\n", averageError);
      System.out.printf("Training Time: %.0f milliseconds\n", (endTime - startTime));
      System.out.print("Reason: ");
      
      if (averageError <= errorThreshold)
      {
         System.out.println("Error threshold reached.");
      }
      else
      {
         System.out.println("Maximum iterations reached.");
      }
      
      if (printNetworkSpecifics)
      {
         printNetworkWeights();
      }
   } // public void printTrainResults()

/**
 * Runs the network for all test cases, calculating the hidden activations and output for each case.
 */
   public void runAll()
   {
      startTimer();

      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         setUpTestCase(caseIndex);
         runByCase(caseIndex);
      }

      endTimer();
   } // public void runAll()

/**
 * Runs the network for a specific test case index, calculating the hidden activations and output.
 * This does not save the theta values, as they are only needed during training.
 * @param caseIndex  the index of the test case to run
 */
   public void runByCase(int caseIndex)
   {
      for (int n = FIRST_H_LAYER_INDEX; n < numActivationLayers; n++)
      {
         for (int j = 0; j < numActivations[n]; j++)
         {
            double theta = 0.0;

            for (int k = 0; k < numActivations[n-1]; k++)
            {
               theta += a[n-1][k] * weights[n-1][k][j];
            }

            a[n][j] = activationFunction(theta);
         } // for (int j = 0; j < numActivations[n]; j++)
      } // for (int n = FIRST_H_LAYER_INDEX; n < numActivationLayers; n++)
   } // public void runByCase(int caseIndex)

/**
 * Sets up the input activations for a specific test case index.
 * @param caseIndex the index of the test case to set up
 */
   public void setUpTestCase(int caseIndex)
   {
      for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
      {
         a[INPUT_LAYER_INDEX][m] = testCaseInput[caseIndex][m];
      }
   }

/**
 * Prints the run results, including optionally the network specifics and truth table.
 */
   public void printRunResults()
   {
      System.out.println("\n---------RUN RESULTS---------");
      System.out.printf("Run Time: %.0f milliseconds\n", (endTime - startTime));

      if (printNetworkSpecifics)
      {
         printNetworkWeights();
      }

      if (printTruthTable)
      {
         printTruthTableWithOutputs();
      }
      else
      {
         printInputOutputOnly();
      }
   } // public void printRunResults()

/**
 * Prints the network's weights from the input layer to the hidden layer and from the hidden layer to the output.
 */
   public void printNetworkWeights()
   {
      System.out.println("\n---------NETWORK WEIGHTS---------");

      for (int n = 0; n < numConnectivityLayers; n++)
      {
         for (int k = 0; k < numActivations[n]; k++)
         {
            for (int j = 0; j < numActivations[n+1]; j++)
            {
               System.out.printf("weights[%d][%d][%d]: %.4f\n", n, k, j, weights[n][k][j]);
            }
         }
      }
   } // public void printNetworkWeights()
   
/**
 * Prints the input table showing input activations for all test cases.
 */
   public void printInputTable()
   {
      System.out.println("\n---------INPUT TABLE---------");
      System.out.println("Inputs");

      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         System.out.print("[");

         for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][m]);
         }
         System.out.println("]");
      }
   } // public void printInputTable()

/**
 * Prints the truth table showing input activations, expected output, and actual output for all test cases.
 * It runs the network for each test case to get the actual outputs.
 */
   public void printTruthTableWithOutputs()
   {
      System.out.println("\n---------TRUTH TABLE---------");
      System.out.println("Inputs | Expected Outputs | Actual Outputs");

      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         setUpTestCase(caseIndex);
         runByCase(caseIndex);
         System.out.print("[");

         for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][m]);
         }

         System.out.print("|");

         for (int i = 0; i < numActivations[outputLayerIndex]; i++)
         {
            System.out.printf(" %.2f", testCaseOutput[caseIndex][i]);
         }

         System.out.print(" |");

         for (int i = 0; i < numActivations[outputLayerIndex]; i++)
         {
            System.out.printf(" %.4f", a[outputLayerIndex][i]);
         }
         
         System.out.println("]");
      } // for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
   } // public void printTruthTableWithOutputs()

/**
 * Prints the input activations and actual outputs for all test cases.
 * It runs the network for each test case to get the actual outputs.
 */
   public void printInputOutputOnly()
   {
      System.out.println("\n---------INPUTS AND OUTPUTS---------");
      System.out.println("Inputs | Outputs");

      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         setUpTestCase(caseIndex);
         runByCase(caseIndex);

         System.out.print("[");

         for (int m = 0; m < numActivations[INPUT_LAYER_INDEX]; m++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][m]);
         }

         System.out.print("|");

         for (int i = 0; i < numActivations[outputLayerIndex]; i++)
         {
            System.out.printf(" %.17f", a[outputLayerIndex][i]);
         }
         
         System.out.println("]");
      } // for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
   } // public void printInputOutputOnly()

/**
 * Prints the activations of the hidden layer.
 */
   public void printHiddenActivations()
   {
      System.out.println("\n---------HIDDEN ACTIVATIONS---------");
      
      for (int n = FIRST_H_LAYER_INDEX; n < outputLayerIndex; n++)
      {
         for (int k = 0; k < numActivations[n]; k++)
         {
            System.out.printf("a[%d][%d]: %.4f\n", n, k, a[n][k]);
         }
      }
   } // public void printHiddenActivations()

/**
 * Saves the current weights of the network to a specified binary file path.
 * The file will first contain the network configuration (A-B-C-D where A, B, C, and D
 * are numbers corresponding to the configuration), followed by the weights in the 
 * appropriate order as doubles.
 * If the file cannot be written to, an exception is thrown.
 */
   public void saveWeightsToFile()
   {
      try
      {
         OutputStream outputStream = new FileOutputStream(saveWeightsFilePath);
         DataOutputStream dataOutputStream = new DataOutputStream(outputStream);

         dataOutputStream.writeUTF(networkConfigString);

         for (int n = 0; n < numConnectivityLayers; n++)
         {
            for (int k = 0; k < numActivations[n]; k++)
            {
               for (int j = 0; j < numActivations[n+1]; j++)
               {
                  dataOutputStream.writeDouble(weights[n][k][j]);
               }
            }
         }
         outputStream.close();
      } // try
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + saveWeightsFilePath);
      }
   } // saveWeightsToFile()

/**
 * Starts the timer by recording the current system time in milliseconds.
 */
   public void startTimer()
   {
      startTime = System.currentTimeMillis();
   }

/**
 * Ends the timer by recording the current system time in milliseconds.
 */
   public void endTimer()
   {
      endTime = System.currentTimeMillis();
   }
} // public class Network
