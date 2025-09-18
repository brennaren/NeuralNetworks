/**
 * Network class that implements a simple A-B-1 feedforward neural network with one activation layer,
 * one hidden layer, and one output. It includes methods to set configurations, allocate memory, 
 * populate the network with weights, and train or run the network.
 * 
 * @author Brenna Ren
 * @version September 17, 2025
 * Date of creation: September 9, 2025
 */
public class Network 
{
   public int numActivationsA;      // number of input activations
   public int numActivationsH;      // number of hidden activations
   public int numOutputsF;          // number of outputs (always 1 for A-B-1 network)

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

   public int numTestCases;         // number of test cases
   
   private double[] a;              // input activations (see design document)
   private double[] h;              // hidden activations (see design document)
   private double[] F0;             // single output for all test cases (see design document)
   private double[][] ah_weights;   // weights from the input layer to hidden layer
   private double[] hF0_weights;    // weights from hidden layer to output
   private double[] h_thetas;       // theta values for the hidden layer that are calculated while finding the weight deltas
   private double F0_theta;         // theta value for the output that is calculated while finding the weight deltas 
   private double[] h_omegas;       // omega values for the hidden layer that are calculated while finding the weight deltas
   private double F0_omega;         // omega value for the output that is calculated while finding the weight deltas 
   private double[] h_psis;         // psi values for the hidden layer that are calculated while finding the weight deltas
   private double F0_psi;           // psi value for the output that is calculated while finding the weight deltas
   
   private double[][] ah_dEdW;   // partial derivatives of the error with respect to weights from input layer to hidden layer
   private double[] hF0_dEdW;    // partial derivatives of the error with respect to weights from hidden layer to output

   private double[][] ah_deltaWeights; // changes in weights from input layer to hidden layer
   private double[] hF0_deltaWeights;  // changes in weights from hidden layer to output
   
   private double averageError;  // average error across all test cases
   private int iteration;        // current training iteration

   private double[][] testCaseInput;   // input values for all test cases
   private double[] testCaseOutput;    // expected output values for all test cases

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
      this.numOutputsF = 1;

      this.randomWeightMin = -1.5;
      this.randomWeightMax = 1.5;
      this.maxIterations = 100000;
      this.errorThreshold = 0.0002;
      this.lambdaValue = 0.3;
      
      this.printNetworkSpecifics = false;
      this.printInputTable = true;
      this.printTruthTable = true;
      this.printHiddenActivations = false;
      
      this.weightConfig = "Random"; // "Manual" or "Random"
      this.isTraining = true;
      this.runAfterTraining = true;

      this.numTestCases = 4;
   } // public void setManualConfigs()

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
      F0 = new double[numTestCases];
      ah_weights = new double[numActivationsA][numActivationsH];
      hF0_weights = new double[numActivationsH];
      h_thetas = new double[numActivationsH];
      testCaseInput = new double[numTestCases][numActivationsA];
      
      if (printTruthTable)
      {
         testCaseOutput = new double[numTestCases];
      }

      if (isTraining)
      {
         h_omegas = new double[numActivationsH];
         h_psis = new double[numActivationsH];
         ah_deltaWeights = new double[numActivationsA][numActivationsH];
         hF0_deltaWeights = new double[numActivationsH];
         ah_dEdW = new double[numActivationsA][numActivationsH];
         hF0_dEdW = new double[numActivationsH];
      } // if (isTraining)
   } // public void allocateNetworkMemory()

   /**
    * Populates the network's weights either with manually specified weights (for testing)
    * or with random weights within the specified range.
    */
   public void populateNetwork()
   {
      if (weightConfig.equals("Manual"))
      {
         fillManualWeights();
      }
      else
      {
         fillRandomWeights();
      }
      fillTestCases();
   } // public void populateNetwork()

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
      hF0_weights[0] = 0.5;
      hF0_weights[1] = 0.6;
   }
   
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
         hF0_weights[j] = getRandomValue(randomWeightMin, randomWeightMax);
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
    * Manually populates the test cases for the network.
    * These values can be changed by modifying this method.
    * It is currently set up for 2 input activations.
    */
   public void fillTestCases()
   {
      testCaseInput[0][0] = 0.0;
      testCaseInput[0][1] = 0.0;
      testCaseOutput[0] = 0.0;

      testCaseInput[1][0] = 0.0;
      testCaseInput[1][1] = 1.0;
      testCaseOutput[1] = 1.0;

      testCaseInput[2][0] = 1.0;
      testCaseInput[2][1] = 0.0;
      testCaseOutput[2] = 1.0;

      testCaseInput[3][0] = 1.0;
      testCaseInput[3][1] = 1.0;
      testCaseOutput[3] = 0.0;
   } // public void fillTestCases()

   /**
    * Trains the network using all training data until the average error is below the error threshold
    * or the maximum number of iterations is reached.
    */
   public void trainAll()
   {
      while (averageError > errorThreshold && iteration < maxIterations)
      {
         for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
         {
            trainByCase(caseIndex);
         }
         iteration++;
         updateAverageError();
      }
   } // public void trainAll()

   /**
    * Updates the average error across all test cases by calculating the error for each case
    * and finding the mean of them.
    */
   public void updateAverageError()
   {
      double totalError = 0.0;
      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         totalError += errorFunction(caseIndex);
      }
      averageError = totalError / numTestCases;
   }

   /**
    * Calculates the error for a specific test case index using the formula:
    * E = 0.5 * (target - output)^2
    * @param caseIndex the index of the test case to use
    * @return the calculated error as a double
    */
   public double errorFunction(int caseIndex)
   {
      return 0.5 * (testCaseOutput[caseIndex] - F0[caseIndex]) * (testCaseOutput[caseIndex] - F0[caseIndex]);
   }

   /**
    * Trains the network for a specific test case index by running the network, updating the delta weights,
    * and then updating the weights.
    * @param caseIndex the index of the test case to use
    */
   public void trainByCase(int caseIndex)
   {
      setUpTestCase(caseIndex);
      runByCase(caseIndex);
      updateDeltaWeights(caseIndex);
      updateWeights();
   }

   /**
    * Updates the network's weights by adding the calculated delta weights to the current weights.
    */
   public void updateWeights()
   {
      for (int k = 0; k < numActivationsA; k++)
      {
         for (int j = 0; j < numActivationsH; j++)
         {
            ah_weights[k][j] += ah_deltaWeights[k][j];
         }
      }

      for (int j = 0; j < numActivationsH; j++)
      {
         hF0_weights[j] += hF0_deltaWeights[j];
      }
   } // public void updateWeights()

   /**
    * Calculates the delta weights for the network based on the output error for a specific test case.
    * Uses the calculations outlined in the design document. Updates the omegas, psis, dEdWs, and deltaWeights. 
    * @param caseIndex the index of the test case to use
    */
   public void updateDeltaWeights(int caseIndex)
   {
      F0_omega = testCaseOutput[caseIndex] - F0[caseIndex];
      F0_psi = F0_omega * derivActivationFunction(F0_theta);

      for (int j = 0; j < numActivationsH; j++)
      {
         hF0_dEdW[j] = -h[j] * F0_psi;
         hF0_deltaWeights[j] = -lambdaValue * hF0_dEdW[j];
      }

      for (int j = 0; j < numActivationsH; j++)
      {
         h_omegas[j] = F0_psi * hF0_weights[j];
         h_psis[j] = h_omegas[j] * derivActivationFunction(h_thetas[j]);
      }

      for (int k = 0; k < numActivationsA; k++)
      {
         for (int j = 0; j < numActivationsH; j++)
         {
            ah_dEdW[k][j] = -a[k] * h_psis[j];
            ah_deltaWeights[k][j] = -lambdaValue * ah_dEdW[k][j];
         }
      }
   } // public void updateDeltaWeights(int caseIndex)

   /**
    * Calculates the derivative of the activation function.
    * Currently, this is the derivative of the sigmoid activation function (f'(theta) = f(theta) * (1 - f(theta))).
    * This can be modified to implement different activation functions as needed.
    * @param theta the input value to the derivative of the activation function
    * @return the output of the derivative of the activation function
    */
   public double derivActivationFunction(double theta)
   {
      double activationFunctionValue = activationFunction(theta);
      return activationFunctionValue * (1.0 - activationFunctionValue);
   }

   /**
    * Prints the training results, including the number of iterations, final average error,
    * and optionally the network specifics.
    */
   public void printTrainResults()
   {
      System.out.println("\n---------TRAINING RESULTS---------");
      System.out.println("Iterations: " + iteration);
      System.out.printf("Final Average Error: %.6f\n", averageError);
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
    * @param caseIndex  the index of the test case to run
    */
   public void runByCase(int caseIndex)
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

      F0_theta = 0.0;
      for (int j = 0; j < numActivationsH; j++)
      {
         F0_theta += h[j] * hF0_weights[j];
      }
      F0[caseIndex] = activationFunction(F0_theta);
   } // public void runByCase(int caseIndex)

   /**
    * Applies the activation function to the given theta value.
    * Currently, this is a sigmoid activation function (f(theta) = 1 / (1 + e^(-theta))).
    * This can be modified to implement different activation functions as needed.
    * @param theta the input value to the activation function
    * @return the output of the activation function
    */
   public double activationFunction(double theta)
   {
      return 1.0 / (1.0 + Math.exp(-theta));
   }

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
         System.out.printf("hF0_weights[%d]: %.4f\n", j, hF0_weights[j]);
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
    */
   public void printTruthTableWithOutputs()
   {
      System.out.println("\n---------TRUTH TABLE---------");
      System.out.println("Inputs | Expected Output | Actual Output");
      for (int caseIndex = 0; caseIndex < numTestCases; caseIndex++)
      {
         System.out.print("[");
         for (int k = 0; k < numActivationsA; k++)
         {
            System.out.printf("%.2f ", testCaseInput[caseIndex][k]);
         }
         System.out.printf("] | %.2f | %.6f\n", testCaseOutput[caseIndex], F0[caseIndex]);
      }
   } // public void printTruthTableWithOutputs()

   /**
    * Prints the activations of the hidden layer.
    */
   public void printHiddenActivations()
   {
      System.out.println("\n---------HIDDEN ACTIVATIONS---------");
      for (int j = 0; j < numActivationsH; j++)
      {
         System.out.printf("h_activations[%d]: %.6f\n", j, h[j]);
      }
   }
} // public class Network
