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
 * Network class that implements an A-B-C-D feed forward neural network with an input activation layer,
 * two hidden layers, and an output layer. It includes methods to set configurations, allocate memory, 
 * populate the network with weights, train or run the network, and output the results. The network will 
 * be trained using gradient descent and with back propagation.
 * 
 * @author Brenna Ren
 * @version November 3, 2025
 * Date of creation: September 9, 2025
 */
public class Network 
{
   public final static String DEFAULT_CONFIG_FILE_PATH = "defaultConfigs.properties";  // default config file path

   public final static int NUM_ACTIVATION_LAYERS = 4; // number of activation layers (4 for A-B-C-D)

   public final static int INPUT_LAYER = 0;     // input layer index
   public final static int FIRST_H_LAYER = 1;   // first hidden layer index
   public final static int SECOND_H_LAYER = 2;  // second hidden layer index
   public final static int OUTPUT_LAYER = 3;    // output layer index

   public int numActivationsA;      // number of input activations
   public int numActivationsH1;     // number of hidden activations (first hidden layer)
   public int numActivationsH2;     // number of hidden activations (second hidden layer)
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
   
   private double[][] a;            // input activations
   private double[][] ah_weights;   // weights from the input layer to hidden layer
   private double[][] h1h2_weights; // weights from hidden layer 1 to hidden layer 2
   private double[][] hF_weights;   // weights from hidden layer to outputs
   private double[] h1_thetas;      // theta values for the hidden layer that are calculated while finding the weight deltas
   private double[] h2_thetas;      // theta values for the hidden layer that are calculated while finding the weight deltas
   private double[] h2_psis;        // psi values for the second hidden layer that are calculated while finding the weight deltas
   private double[] F_psis;         // psi values for the outputs that are calculated while finding the weight deltas

   public String configFilePath;       // file path to load configurations from
   public String loadWeightsFilePath;  // file path to load weights from (binary path)
   public String saveWeightsFilePath;  // file path to save weights to (binary path)
   public String inputsFilePath;       //  file path to load test cases from
   public String outputsFilePath;      // file path to load expected outputs from
   
   private double averageError;  // average error across all test cases
   private int iteration;        // current training iteration

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
 * Initializes the configurations of the Network with default, hard-coded values.
 * These values can be changed by modifying this method.
 */
   public void setManualConfigs()
   {
      this.numActivationsA = 2;
      this.numActivationsH1 = 1;
      this.numActivationsH2 = 1;
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

         this.numActivationsA = Integer.parseInt(props.getProperty("numActivationsA"));
         this.numActivationsH1 = Integer.parseInt(props.getProperty("numActivationsH1"));
         this.numActivationsH2 = Integer.parseInt(props.getProperty("numActivationsH2"));
         this.numOutputsF = Integer.parseInt(props.getProperty("numOutputsF"));

         this.randomWeightMin = Double.parseDouble(props.getProperty("randomWeightMin"));
         this.randomWeightMax = Double.parseDouble(props.getProperty("randomWeightMax"));
         this.maxIterations = Integer.parseInt(props.getProperty("maxIterations"));
         this.errorThreshold = Double.parseDouble(props.getProperty("errorThreshold"));
         this.lambdaValue = Double.parseDouble(props.getProperty("lambdaValue"));

         this.printNetworkSpecifics = Boolean.parseBoolean(props.getProperty("printNetworkSpecifics"));
         this.printInputTable = Boolean.parseBoolean(props.getProperty("printInputTable"));
         this.printTruthTable = Boolean.parseBoolean(props.getProperty("printTruthTable"));
         this.printHiddenActivations = Boolean.parseBoolean(props.getProperty("printHiddenActivations"));

         this.weightConfig = props.getProperty("weightConfig");
         this.loadWeightsFilePath = props.getProperty("loadWeightsFilePath");
         this.saveWeightsFilePath = props.getProperty("saveWeightsFilePath");
         this.saveWeightsToFile = Boolean.parseBoolean(props.getProperty("saveWeightsToFile"));

         this.isTraining = Boolean.parseBoolean(props.getProperty("isTraining"));
         this.runAfterTraining = Boolean.parseBoolean(props.getProperty("runAfterTraining"));

         this.testCaseConfig = props.getProperty("testCaseConfig");
         this.numTestCases = Integer.parseInt(props.getProperty("numTestCases"));
         this.inputsFilePath = props.getProperty("inputsFilePath");
         this.outputsFilePath = props.getProperty("outputsFilePath");
      } // try
      catch (Exception e)
      {
         throw new IllegalArgumentException("Error: Unable to open file at " + configFilePath);
      }
   } // public void loadConfigsFromFile()

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
 * The expected format is the network configuration, followed by the weights 
 * in the appropriate order as doubles.
 */
   public void loadWeightsFromFile()
   {
      try 
      {
         InputStream inputStream = new FileInputStream(loadWeightsFilePath);
         DataInput dataInputStream = new DataInputStream(inputStream);

         String fileNetworkConfig = dataInputStream.readUTF();
         String actualNetworkConfig = numActivationsA + "-" + numActivationsH1 + "-" + numActivationsH2 + "-" + numOutputsF;

         if (!fileNetworkConfig.equals(actualNetworkConfig))
         {
            inputStream.close();
            throw new IllegalArgumentException("Error: Weight configuration in file does not match network configuration.");
         }

         for (int m = 0; m < numActivationsA; m++)
         {
            for (int k = 0; k < numActivationsH1; k++)
            {
               ah_weights[m][k] = dataInputStream.readDouble();
            }
         }

         for (int k = 0; k < numActivationsH1; k++)
         {
            for (int j = 0; j < numActivationsH2; j++)
            {
               h1h2_weights[k][j] = dataInputStream.readDouble();
            }
         }

         for (int j = 0; j < numActivationsH2; j++)
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
      System.out.println("Configurations File Path: " + configFilePath);
      System.out.println("Type of Network: " + numActivationsA + "-" + numActivationsH1 + "-" + numActivationsH2 + "-" + numOutputsF);
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
      a = new double[NUM_ACTIVATION_LAYERS][];
      a[INPUT_LAYER] = new double[numActivationsA];
      a[FIRST_H_LAYER] = new double[numActivationsH1];
      a[SECOND_H_LAYER] = new double[numActivationsH2];
      a[OUTPUT_LAYER] = new double[numOutputsF];
      ah_weights = new double[numActivationsA][numActivationsH1];
      h1h2_weights = new double[numActivationsH1][numActivationsH2];
      hF_weights = new double[numActivationsH2][numOutputsF];
      testCaseInput = new double[numTestCases][numActivationsA];
      
      if (isTraining || printTruthTable)
      {
         testCaseOutput = new double[numTestCases][numOutputsF];
      }

      if (isTraining)
      {
         h1_thetas = new double[numActivationsH1];
         h2_thetas = new double[numActivationsH2];
         h2_psis = new double[numActivationsH2];
         F_psis = new double[numOutputsF];
      }
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
      for (int m = 0; m < numActivationsA; m++)
      {
         for (int k = 0; k < numActivationsH1; k++)
         {
            ah_weights[m][k] = getRandomValue(randomWeightMin, randomWeightMax);
         }
      }

      for (int k = 0; k < numActivationsH1; k++)
      {
         for (int j = 0; j < numActivationsH2; j++)
         {
            h1h2_weights[k][j] = getRandomValue(randomWeightMin, randomWeightMax);
         }
      }

      for (int j = 0; j < numActivationsH2; j++)
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
         for (int m = 0; m < numActivationsA; m++)
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
         } // for (int m = 0; m < numActivationsA; m++)
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
            for (int i = 0; i < numOutputsF; i++)
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
            } // for (int i = 0; i < numOutputsF; i++)
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
      } // while (averageError > errorThreshold && iteration < maxIterations)
   } // public void trainAll()


/**
 * Updates the weights for the network based on the calculated psis from the outputs and the activations.
 * Uses the calculations outlined in the design document. Stores only the psis and weights.
 * @param caseIndex the index of the test case to use
 */
   public void updateWeights(int caseIndex)
   {
      for (int j = 0; j < numActivationsH2; j++)
      {
         double h2_omega = 0.0;

         for (int i = 0; i < numOutputsF; i++)
         {
            h2_omega += F_psis[i] * hF_weights[j][i];
            hF_weights[j][i] += lambdaValue * a[SECOND_H_LAYER][j] * F_psis[i];
         }
         
         h2_psis[j] = h2_omega * derivActivationFunction(h2_thetas[j]);
      } // for (int j = 0; j < numActivationsH2; j++)

      for (int k = 0; k < numActivationsH1; k++)
      {
         double h1_omega = 0.0;

         for (int j = 0; j < numActivationsH2; j++)
         {
            h1_omega += h2_psis[j] * h1h2_weights[k][j];
            h1h2_weights[k][j] += lambdaValue * a[FIRST_H_LAYER][k] * h2_psis[j];
         }

         double h1_psi = h1_omega * derivActivationFunction(h1_thetas[k]);

         for (int m = 0; m < numActivationsA; m++)
         {
            ah_weights[m][k] += lambdaValue * a[INPUT_LAYER][m] * h1_psi;
         }
      } // for (int k = 0; k < numActivationsH1; k++)
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
      for (int k = 0; k < numActivationsH1; k++)
      {
         h1_thetas[k] = 0.0;

         for (int m = 0; m < numActivationsA; m++)
         {
            h1_thetas[k] += a[INPUT_LAYER][m] * ah_weights[m][k];
         }

         a[FIRST_H_LAYER][k] = activationFunction(h1_thetas[k]);
      } // for (int k = 0; k < numActivationsH1; k++)

      for (int j = 0; j < numActivationsH2; j++)
      {
         h2_thetas[j] = 0.0;

         for (int k = 0; k < numActivationsH1; k++)
         {
            h2_thetas[j] += a[FIRST_H_LAYER][k] * h1h2_weights[k][j];
         }

         a[SECOND_H_LAYER][j] = activationFunction(h2_thetas[j]);
      } // for (int j = 0; j < numActivationsH2; j++)

      for (int i = 0; i < numOutputsF; i++)
      {
         double F_theta = 0.0;

         for (int j = 0; j < numActivationsH2; j++)
         {
            F_theta += a[SECOND_H_LAYER][j] * hF_weights[j][i];
         }

         a[OUTPUT_LAYER][i] = activationFunction(F_theta);
         double F_omega = testCaseOutput[caseIndex][i] - a[OUTPUT_LAYER][i];
         F_psis[i] = F_omega * derivActivationFunction(F_theta);
         error += F_omega * F_omega;
      } // for (int i = 0; i < numOutputsF; i++)

      return error;
   } // public void runByCase(int caseIndex)

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
      for (int k = 0; k < numActivationsH1; k++)
      {
         double h1_theta = 0.0;

         for (int m = 0; m < numActivationsA; m++)
         {
            h1_theta += a[INPUT_LAYER][m] * ah_weights[m][k];
         }

         a[FIRST_H_LAYER][k] = activationFunction(h1_theta);
      } // for (int k = 0; k < numActivationsH1; k++)

      for (int j = 0; j < numActivationsH2; j++)
      {
         double h2_theta = 0.0;

         for (int k = 0; k < numActivationsH1; k++)
         {
            h2_theta += a[FIRST_H_LAYER][k] * h1h2_weights[k][j];
         }

         a[SECOND_H_LAYER][j] = activationFunction(h2_theta);
      } // for (int j = 0; j < numActivationsH2; j++)

      for (int i = 0; i < numOutputsF; i++)
      {
         double F_theta = 0.0;

         for (int j = 0; j < numActivationsH2; j++)
         {
            F_theta += a[SECOND_H_LAYER][j] * hF_weights[j][i];
         }
         
         a[OUTPUT_LAYER][i] = activationFunction(F_theta);
      } // for (int i = 0; i < numOutputsF; i++)
   } // public void runByCase(int caseIndex)

/**
 * Sets up the input activations for a specific test case index.
 * @param caseIndex the index of the test case to set up
 */
   public void setUpTestCase(int caseIndex)
   {
      for (int m = 0; m < numActivationsA; m++)
      {
         a[INPUT_LAYER][m] = testCaseInput[caseIndex][m];
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
      System.out.println("Weights from Input Layer to First Hidden Layer (ah_weights):");

      for (int m = 0; m < numActivationsA; m++)
      {
         for (int k = 0; k < numActivationsH1; k++)
         {
            System.out.printf("ah_weights[%d][%d]: %.4f\n", m, k, ah_weights[m][k]);
         }
      }
      
      System.out.println("Weights from First Hidden Layer to Second Hidden Layer (h1h2_weights):");
      
      for (int k = 0; k < numActivationsH1; k++)
      {
         for (int j = 0; j < numActivationsH2; j++)
         {
            System.out.printf("h1h2_weights[%d][%d]: %.4f\n", k, j, h1h2_weights[k][j]);
         }
      }

      System.out.println("Weights from Second Hidden Layer to Output (hF_weights):");

      for (int j = 0; j < numActivationsH2; j++)
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

         for (int m = 0; m < numActivationsA; m++)
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

         for (int m = 0; m < numActivationsA; m++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][m]);
         }

         System.out.print("|");

         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf(" %.2f", testCaseOutput[caseIndex][i]);
         }

         System.out.print(" |");

         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf(" %.4f", a[OUTPUT_LAYER][i]);
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

         for (int m = 0; m < numActivationsA; m++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][m]);
         }

         System.out.print("|");

         for (int i = 0; i < numOutputsF; i++)
         {
            System.out.printf(" %.17f", a[OUTPUT_LAYER][i]);
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
      
      for (int k = 0; k < numActivationsH1; k++)
      {
         System.out.printf("h_activations[%d]: %.4f\n", k, a[FIRST_H_LAYER][k]);
      }

      for (int j = 0; j < numActivationsH2; j++)
      {
         System.out.printf("h_activations[%d]: %.4f\n", j, a[SECOND_H_LAYER][j]);
      }
   } // public void printHiddenActivations()

/**
 * Saves the current weights of the network to a specified binary file path.
 * The file will first contain the network configuration (A-B-C where A, B, and C 
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

         String networkConfig = numActivationsA + "-" + numActivationsH1 + "-"  + numActivationsH2 + "-"+ numOutputsF;
         dataOutputStream.writeUTF(networkConfig);

         for (int m = 0; m < numActivationsA; m++)
         {
            for (int k = 0; k < numActivationsH1; k++)
            {
               dataOutputStream.writeDouble(ah_weights[m][k]);
            }
         }

         for (int k = 0; k < numActivationsH1; k++)
         {
            for (int j = 0; j < numActivationsH2; j++)
            {
               dataOutputStream.writeDouble(h1h2_weights[k][j]);
            }
         }
         
         for (int j = 0; j < numActivationsH2; j++)
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
