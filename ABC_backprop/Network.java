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
 * Network class that implements an A-B-C feedforward neural network with one activation layer,
 * one hidden layer, and an output layer. It includes methods to set configurations, allocate memory, 
 * populate the network with weights, and train or run the network. The network will be trained using
 * gradient descent and with back propagation.
 * 
 * @author Brenna Ren
 * @version October 15, 2025
 * Date of creation: September 9, 2025
 */
public class Network 
{
   public final static String DEFAULT_CONFIG_FILE_PATH = "defaultConfigs.json"; // default config file path

   public int numActivationsA;      // number of input activations
   public int numActivationsH;      // number of hidden activations
   public int numOutputsF;          // number of outputs

   public double randomWeightMin;   // minimum random weight value
   public double randomWeightMax;   // maximum random weight value
   public int maxIterations;        // maximum number of training iterations before stopping
   public double errorThreshold;    // training stops when average error is below this threshold
   public double lambdaValue;       // learning factor used to control the magnitude of weight updates

   public boolean printNetworkSpecifics;  // whether to print network specifics after training/running
   public boolean printInputTable;        // whether to print the input table after training/running
   public boolean printTruthTable;        // whether to print the truth table after training/running
   public boolean printHiddenActivations; // whether to print the hidden activations after each run

   public String weightConfig;         // whether to use manually specified weights or random weights
   public boolean isTraining;          // whether the network is in training mode (true) or running mode (false)
   public boolean runAfterTraining;    // whether to run the network after training
   public boolean saveWeightsToFile;   // whether to save weights to a file after training

   public String testCaseConfig;    // whether to use manually specified test cases or load from a file
   public int numTestCases;         // number of test cases
   
   private double[] a;              // input activations (see design document)
   private double[] h;              // hidden activations (see design document)
   private double[] F;              // outputs (see design document)
   private double[][] ah_weights;   // weights from the input layer to hidden layer
   private double[][] hF_weights;   // weights from hidden layer to outputs
   private double[] h_thetas;       // theta values for the hidden layer that are calculated while finding the weight deltas
   private double[] F_thetas;       // theta values for the output that are calculated while finding the weight deltas 
   private double[] F_psis;         // psi values for the outputs that are calculated while finding the weight deltas

   public String configFilePath;       // file path to load configurations from
   private String loadWeightsFilePath; // file path to load weights from (binary path)
   private String saveWeightsFilePath; // file path to save weights to (binary path)
   private String testCasesFilePath;   //  file path to load test cases from
   
   private double totalError;    // total error across all test cases
   private double averageError;  // average error across all test cases
   private int iteration;        // current training iteration

   private double[][] testCaseInput;   // input values for all test cases
   private double[][] testCaseOutput;  // expected output values for all test cases

/**
 * Initializes the network's variables to default values.
 */
   public void initializeVariables()
   {
      averageError = Double.MAX_VALUE;
      iteration = 0;
   }

/**
 * Initializes the configurations of the Network with default, hard-coded values.
 * These values can be changed by modifying this method.
 */
   public void setManualConfigs()
   {
      this.numActivationsA = 2;
      this.numActivationsH = 1;
      this.numOutputsF = 3;

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

      this.testCaseConfig = "File"; // "Manual" or "File"
      this.numTestCases = 4;
      this.testCasesFilePath = "AND_OR_XOR/AND_OR_XOR_test_cases.txt";
   } // public void setManualConfigs()

/**
 * Loads the network configurations from a specified JSON file path.
 * If the file cannot be read, an exception is thrown.
 * The expected format is JSON with keys matching the configuration variable names.
 */
   public void loadConfigsFromFile()
   {
      //TODO: Implement JSON parsing to load configurations from file
   }
/**
 * Fills the weights array with manually specified weights.
 * These values can be changed by modifying this method.
 */
   public void fillManualWeights()
   {
      ah_weights[0][0] = 0.1;
      ah_weights[1][0] = 0.2;
      ah_weights[0][1] = 0.3;
      ah_weights[1][1] = 0.4;
      hF_weights[0][0] = 0.5;
      hF_weights[1][0] = 0.6;
   }

/**
 * Fills the test cases array with manually specified test cases.
 * These values can be changed by modifying this method.
 */
   public void fillManualTestCases()
   {
      testCaseInput[0][0] = 0.0;
      testCaseInput[0][1] = 0.0;
      testCaseInput[1][0] = 0.0;
      testCaseInput[1][1] = 1.0;
      testCaseInput[2][0] = 1.0;
      testCaseInput[2][1] = 0.0;
      testCaseInput[3][0] = 1.0;
      testCaseInput[3][1] = 1.0;

      testCaseOutput[0][0] = 0.0; // AND
      testCaseOutput[0][1] = 0.0; // OR
      testCaseOutput[1][0] = 0.0; // AND
      testCaseOutput[1][1] = 1.0; // OR
      testCaseOutput[2][0] = 0.0; // AND
      testCaseOutput[2][1] = 1.0; // OR
      testCaseOutput[3][0] = 1.0; // AND
      testCaseOutput[3][1] = 1.0; // OR
   }

/**
 * Loads weights from a specified binary file path into the network's weight arrays.
 * If the file cannot be read, an exception is thrown.
 * The expected format is the weights in the appropriate order as doubles.
 */
   public void loadWeightsFromFile()
   {
      try 
      {
         InputStream inputStream = new FileInputStream(loadWeightsFilePath);
         DataInput dataInputStream = new DataInputStream(inputStream);
         for (int k = 0; k < numActivationsA; k++)
         {
            for (int j = 0; j < numActivationsH; j++)
            {
               ah_weights[k][j] = dataInputStream.readDouble();
            }
         }
         for (int j = 0; j < numActivationsH; j++)
         {
            for (int i = 0; i < numOutputsF; i++)
            {
               hF_weights[j][i] = dataInputStream.readDouble();
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
      System.out.println("Type of Network: " + numActivationsA + "-" + numActivationsH + "-" + numOutputsF);
      System.out.println("Print Network Specifics: " + printNetworkSpecifics);
      System.out.println("Print Input Table: " + printInputTable);
      System.out.println("Print Truth Table: " + printTruthTable);
      System.out.println("Print Hidden Activations: " + printHiddenActivations);
      System.out.println("Weight Configuration: " + weightConfig);
      System.out.println("Test Case Configuration: " + testCaseConfig);
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
 */
   public void allocateNetworkMemory()
   {
      a = new double[numActivationsA];
      h = new double[numActivationsH];
      F = new double[numOutputsF];
      ah_weights = new double[numActivationsA][numActivationsH];
      hF_weights = new double[numActivationsH][numOutputsF];
      testCaseInput = new double[numTestCases][numActivationsA];
      
      if (isTraining || printTruthTable)
      {
         testCaseOutput = new double[numTestCases][numOutputsF];
      }

      if (isTraining)
      {
         h_thetas = new double[numActivationsH];
         F_thetas = new double[numOutputsF];
         F_psis = new double[numOutputsF];
      } // if (isTraining)
   } // public void allocateNetworkMemory()

/**
 * Populates the network's weights either with manually specified weights from a file
 * or with random weights within the specified range.
 */
   public void populateNetwork()
   {
      if (weightConfig.equals("Manual"))
      {
         fillManualWeights();
      }
      else if (weightConfig.equals("Load"))
      {
         loadWeightsFromFile();
      }
      else
      {
         fillRandomWeights();
      }
      if (testCaseConfig.equals("Manual"))
      {
         fillManualTestCases();
      }
      else
      {
         fillFileTestCases();
      }
   } // public void populateNetwork()
   
/**
 * Fills the weights array with randomized weights between randomWeightMin and randomWeightMax.
 */
   public void fillRandomWeights()
   {
      for (int k = 0; k < numActivationsA; k++)
      {
         for (int j = 0; j < numActivationsH; j++)
         {
            ah_weights[k][j] = getRandomValue(randomWeightMin, randomWeightMax);
         }
      }

      for (int j = 0; j < numActivationsH; j++)
      {
         for (int i = 0; i < numOutputsF; i++)
         {
            hF_weights[j][i] = getRandomValue(randomWeightMin, randomWeightMax);
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
      Scanner fileScanner;
      try
      {
         fileScanner = new Scanner(new File(testCasesFilePath));
      }
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + testCasesFilePath);
      }

      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         for (int k = 0; k < numActivationsA; k++)
         {
            if (fileScanner.hasNextDouble())
            {
               testCaseInput[caseIndex][k] = fileScanner.nextDouble();
            }
            else
            {
               fileScanner.close();
               throw new IllegalArgumentException("Error: Not enough input values in test cases file.");
            }
         } // for (int k = 0; k < numActivationsA; k++)

         for (int i = 0; i < numOutputsF; i++)
         {
            if (fileScanner.hasNextDouble())
            {
               if (isTraining || printTruthTable)
               {
                  testCaseOutput[caseIndex][i] = fileScanner.nextDouble();
               }
               else
               {
                  System.out.println(fileScanner.nextDouble());
               }
            } // if (fileScanner.hasNextDouble())
            else
            {
               fileScanner.close();
               throw new IllegalArgumentException("Error: Not enough output values in test cases file.");
            }
         } // for (int i = 0; i < numOutputsF; i++)
      } // for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      fileScanner.close();
   } // public void fillFileTestCases()

/**
 * Trains the network using all training data until the average error is below the error threshold
 * or the maximum number of iterations is reached.
 */
   public void trainAll()
   {
      while (averageError > errorThreshold && iteration < maxIterations)
      {
         totalError = 0.0;

         for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
         {
            setUpTestCase(caseIndex);
            runForTrainByCase(caseIndex);
            updateWeights(caseIndex);
         }

         totalError /= 2.0;
         iteration++;
         averageError = totalError / numTestCases;
      } // while (averageError > errorThreshold && iteration < maxIterations)
   } // public void trainAll()


/**
 * Updates the weights for the network based on the calculated psis from the outputs and the activations.
 * Uses the calculations outlined in the design document. Stores only the psis and weights.
 * @param caseIndex the index of the test case to use
 */
   public void updateWeights(int caseIndex)
   {
      for (int j = 0; j < numActivationsH; j++)
      {
         double h_omega = 0.0;

         for (int i = 0; i < numOutputsF; i++)
         {
            h_omega += F_psis[i] * hF_weights[j][i];
            double hF_deltaWeight = lambdaValue * h[j] * F_psis[i];
            hF_weights[j][i] += hF_deltaWeight;
         }
         
         double h_psi = h_omega * derivActivationFunction(h_thetas[j]);

         for (int k = 0; k < numActivationsA; k++)
         {
            double ah_deltaWeight = lambdaValue * a[k] * h_psi;
            ah_weights[k][j] += ah_deltaWeight;
         }
      } // for (int j = 0; j < numActivationsH; j++)
   } // public void updateWeights(int caseIndex)

/**
 * Calculates the derivative of the activation function.
 * Currently, this is the derivative of the sigmoid activation function.
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
 * Currently, this is calls the sigmoid activation function.
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
 * Runs the network for training for a specific test case index, calculating the hidden activations and output.
 * This saves the theta values, as they are needed during training.
 * @param caseIndex  the index of the test case to run
 */
   public void runForTrainByCase(int caseIndex)
   {
      for (int j = 0; j < numActivationsH; j++)
      {
         h_thetas[j] = 0.0;
         for (int k = 0; k < numActivationsA; k++)
         {
            h_thetas[j] += a[k] * ah_weights[k][j];
         }
         h[j] = activationFunction(h_thetas[j]);
      }

      for (int i = 0; i < numOutputsF; i++)
      {
         F_thetas[i] = 0.0;
         for (int j = 0; j < numActivationsH; j++)
         {
            F_thetas[i] += h[j] * hF_weights[j][i];
         }
         F[i] = activationFunction(F_thetas[i]);
         double F_omega = testCaseOutput[caseIndex][i] - F[i];
         F_psis[i] = F_omega * derivActivationFunction(F_thetas[i]);
         totalError += F_omega * F_omega;
      }
   } // public void runByCase(int caseIndex)

/**
 * Prints the training results, including the number of iterations, final average error,
 * and optionally the network specifics.
 */
   public void printTrainResults()
   {
      System.out.println("\n---------TRAINING RESULTS---------");
      System.out.println("Iterations: " + iteration);
      System.out.printf("Final Average Error: %.4f\n", averageError);
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
      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         setUpTestCase(caseIndex);
         runByCase(caseIndex);
      }
   } // public void runAll()

/**
 * Runs the network for a specific test case index, calculating the hidden activations and output.
 * This does not save the theta values, as they are only needed during training.
 * @param caseIndex  the index of the test case to run
 */
   public void runByCase(int caseIndex)
   {
      for (int j = 0; j < numActivationsH; j++)
      {
         double h_theta = 0.0;
         for (int k = 0; k < numActivationsA; k++)
         {
            h_theta += a[k] * ah_weights[k][j];
         }
         h[j] = activationFunction(h_theta);
      }

      for (int i = 0; i < numOutputsF; i++)
      {
         double F_theta = 0.0;
         for (int j = 0; j < numActivationsH; j++)
         {
            F_theta += h[j] * hF_weights[j][i];
         }
         F[i] = activationFunction(F_theta);
      }
   } // public void runByCase(int caseIndex)

/**
 * Sets up the input activations for a specific test case index.
 * @param caseIndex the index of the test case to set up
 */
   public void setUpTestCase(int caseIndex)
   {
      for (int k = 0; k < numActivationsA; k++)
      {
         a[k] = testCaseInput[caseIndex][k];
      }
   }

/**
 * Prints the run results, including optionally the network specifics and truth table.
 */
   public void printRunResults()
   {
      System.out.println("\n---------RUN RESULTS---------");
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
      System.out.println("Weights from Input Layer to Hidden Layer (ah_weights):");
      for (int k = 0; k < numActivationsA; k++)
      {
         for (int j = 0; j < numActivationsH; j++)
         {
            System.out.printf("ah_weights[%d][%d]: %.4f\n", k, j, ah_weights[k][j]);
         }
      }

      System.out.println("Weights from Hidden Layer to Output (hF0_weights):");
      for (int j = 0; j < numActivationsH; j++)
      {
         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf("hF_weights[%d][%d]: %.4f\n", j, i, hF_weights[j][i]);
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
         for (int k = 0; k < numActivationsA; k++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][k]);
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
         for (int k = 0; k < numActivationsA; k++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][k]);
         }

         System.out.print("|");
         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf(" %.2f", testCaseOutput[caseIndex][i]);
         }

         System.out.print(" |");
         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf(" %.4f", F[i]);
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
         for (int k = 0; k < numActivationsA; k++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][k]);
         }

         System.out.print("|");
         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf(" %.4f", F[i]);
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
      for (int j = 0; j < numActivationsH; j++)
      {
         System.out.printf("h_activations[%d]: %.4f\n", j, h[j]);
      }
   }

/**
 * Saves the current weights of the network to a specified binary file path.
 * The file will contain the weights in the appropriate format.
 * If the file cannot be written to, an exception is thrown.
 */
   public void saveWeightsToFile()
   {
      try
      {
         OutputStream outputStream = new FileOutputStream(saveWeightsFilePath);
         DataOutputStream dataOutputStream = new DataOutputStream(outputStream);
         for (int k = 0; k < numActivationsA; k++)
         {
            for (int j = 0; j < numActivationsH; j++)
            {
               dataOutputStream.writeDouble(ah_weights[k][j]);
            }
         }
         for (int j = 0; j < numActivationsH; j++)
         {
            for (int i = 0; i < numOutputsF; i++)
            {
               dataOutputStream.writeDouble(hF_weights[j][i]);
            }
         }
         outputStream.close();
      } // try
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + saveWeightsFilePath);
      }
   } // saveWeightsToFile()
} // public class Network
