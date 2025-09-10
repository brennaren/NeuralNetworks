/**
 * Network class that implements a simple feedforward neural network with one hidden layer.
 * It includes methods to set configurations, allocate memory, populate the network with weights,
 * and train or run the network.
 * 
 * @author Brenna Ren
 * @version September 9, 2025
 * Date of creation: September 9, 2025
 * 
 * Functions in this class:
 * void setManualConfigs()
 * void printTrainingConfigs()
 * void allocateNetworkMemory()
 * void populateNetwork()
 * void trainAndReport()
 * void runAndReport()
 * void fillManualWeights()
 * void fillRandomWeights()
 * double getRandomWeight()
 */
public class Network {
   public int numActivationsA;      // number of input activations
   public int numActivationsH;      // number of hidden activations

   public double randomWeightMin;   // minimum random weight value
   public double randomWeightMax;   // maximum random weight value

   public int maxIterations;        // maximum number of training iterations before stopping

   public double errorThreshold;    // training stops when average error is below this threshold

   public double lambdaValue;       // learning factor used to control the magnitude of weight updates

   public boolean printNetworkSpecifics;  // whether to print network specifics after training/running
   public boolean printInputTable;        // whether to print the input table after training/running
   public boolean printTruthTable;        // whether to print the truth table after training/running
   public boolean printDebugInfo;         // whether to print debug information during training/running

   public boolean isManualWeights;  // whether to use manually specified weights or random weights

   public boolean isTraining;       // whether the network is in training mode (true) or running mode (false)

   public int numTestCases;         // number of test cases
   
   private double[] a;              // input activations (see design document)
   private double[] h;              // hidden activations (see design document)
   private double F0;               // single output (see design document)
   private double[][] ah_weights;   // weights from the input layer to hidden layer
   private double[] hF0_weights;    // weights from hidden layer to output
   private double[] thetas;         // theta values that are calculated while finding the weight deltas
   private double[] omegas;         // omega values that are calculated while finding the weight deltas
   private double[] psis;           // psi values that are calculated while finding the weight deltas
   
   private double[][] ah_deltaWeights; // changes in weights from input layer to hidden layer
   private double[] hF0_deltaWeights;  // changes in weights from hidden layer to output
   
   private double[][] testCaseInput;   // input values for all test cases
   private double[] testCaseOutput;    // expected output values for all test cases


   /**
    * Initializes the configurations of the Network with default, hard-coded values.
    * These values can be changed by modifying this method.
    */
   public void setManualConfigs()
   {
      this.numActivationsA = 2;
      this.numActivationsH = 2;
      this.randomWeightMin = 0;
      this.randomWeightMax = 1.0;
      this.maxIterations = 1000;
      this.errorThreshold = 0.01;
      this.lambdaValue = 0.3;
      this.printNetworkSpecifics = false;
      this.printDebugInfo = false;
      this.isManualWeights = false;
      this.numTestCases = 4;
   } // public void SetManualConfigs()

   /**
    * Echos the network configurations by printing them to the console.
    */
   public void printNetworkConfigs()
   {
      System.out.println("---------NETWORK CONFIGURATIONS (A-B-1 Network)---------");
      System.out.println("Number of Activations A: " + numActivationsA);
      System.out.println("Number of Activations H: " + numActivationsH);
   }
   
   /**
    * Echos the training parameters by printing them to the console.
    */
   public void printTrainingParameters()
   {
      System.out.println("---------Training Parameters---------");
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
      ah_weights = new double[numActivationsA][numActivationsH];
      hF0_weights = new double[numActivationsH];
      thetas = new double[numActivationsH];
      omegas = new double[numActivationsH];
      psis = new double[numActivationsH];
      ah_deltaWeights = new double[numActivationsA][numActivationsH];
      hF0_deltaWeights = new double[numActivationsH];
      testCaseInput = new double[numTestCases][numActivationsA];
      testCaseOutput = new double[numTestCases];
   } // public void AllocateNetworkMemory()

   /**
    * Populates the network's weights either with manually specified weights (for testing)
    * or with random weights within the specified range.
    */
   public void populateNetwork()
   {
      if (isManualWeights)
      {
         fillManualWeights();
      }
      else
      {
         fillRandomWeights();
      }
      fillTestCases();
   } // public void PopulateNetwork()

   /**
    * Fills the weights array with manually specified weights, primarily for testing against the 
    * spreadsheet from Project #1.
    * These values can be changed by modifying this method.
    */
   public void fillManualWeights()
   {
      ah_weights[0][0] = 0.8;
      ah_weights[1][0] = 0.8;
      ah_weights[0][1] = 0.9;
      ah_weights[1][1] = 0.9;
      hF0_weights[0] = 0.6;
      hF0_weights[1] = 0.2;
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
            ah_weights[k][j] = getRandomWeight();
         }
      }

      for (int j = 0; j < numActivationsH; j++)
      {
         hF0_weights[j] = getRandomWeight();
      }
   } // public void FillRandomWeights()

   /**
    * Generates a random weight between randomWeightMin and randomWeightMax.
    * @return a random weight as a double
    */
   public double getRandomWeight()
   {
      return Math.random() * (randomWeightMax - randomWeightMin) + randomWeightMin;
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
      testCaseOutput[3] = 1.0;
   } // public void FillTestCases()

   public void train()
   {

   }

   public void run()
   {

   }

   public void printNetworkWeights()
   {
      System.out.println("---------NETWORK WEIGHTS---------");
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
   }

   public void printTruthTable()
   {

   }

} // public class Network
