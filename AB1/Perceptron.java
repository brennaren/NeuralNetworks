/**
 * Percetron class that implements a simple feedforward neural network with one hidden layer.
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
public class Perceptron {
   private int numActivationsA;
   private int numActivationsH;

   private int numTestCases;

   private double randomWeightMin;
   private double randomWeightMax;

   private int maxIterations;

   private double errorThreshold;

   private double learningFactor; // AKA lambda value

   private boolean printNetworkSpecifics;
   private boolean printDebugInfo;

   private boolean isManualWeights;

   private double[] a;
   private double[] h;
   private double F0;
   private double[][] weights;
   private double[] thetas;
   private double[] omegas;
   private double[] psis;
   private double[][] deltaWeights;

   private double[][] testCaseInput;
   private double[] testCaseOutput;


   /**
    * Initializes the configurations of the Perceptron with default, hard-coded values.
    * These values can be changed by modifying this method.
    */
   public void setManualConfigs()
   {
      this.numActivationsA = 4;
      this.numActivationsH = 2;
      this.randomWeightMin = 0;
      this.randomWeightMax = 1.0;
      this.maxIterations = 1000;
      this.errorThreshold = 0.01;
      this.learningFactor = 0.1;
      this.printNetworkSpecifics = false;
      this.printDebugInfo = false;
      this.isManualWeights = true;
   }

   /**
    * Echos the training configurations by printing them to the console.
    */
   public void printTrainingConfigs()
   {
      System.out.println("Training Configurations:");
      System.out.println("Number of Activations A: " + numActivationsA);
      System.out.println("Number of Activations H: " + numActivationsH);
      System.out.println("Random Weight Min: " + randomWeightMin);
      System.out.println("Random Weight Max: " + randomWeightMax);
      System.out.println("Max Iterations: " + maxIterations);
      System.out.println("Error Threshold: " + errorThreshold);
      System.out.println("Learning Factor: " + learningFactor);
      System.out.println("Print Network Specifics: " + printNetworkSpecifics);
      System.out.println("Print Debug Info: " + printDebugInfo);
   }

   /**
    * Allocates memory for the network's arrays based on the number of activations.
    */
   public void allocateNetworkMemory()
   {
      a = new double[numActivationsA];
      h = new double[numActivationsH];
      weights = new double[numActivationsA][numActivationsH];
      thetas = new double[numActivationsH];
      omegas = new double[numActivationsH];
      deltaWeights = new double[numActivationsA][numActivationsH];
      testCaseInput = new double[numTestCases][numActivationsA];
      testCaseOutput = new double[numTestCases];
   }

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
   }

   public void trainAndReport()
   {

   }

   public void runAndReport()
   {

   }

   /**
    * Fills the weights array with manually specified weights, primarily for testing against the 
    * spreadsheet from Project #1.
    */
   public void fillManualWeights()
   {
      weights[0][0] = 0.8;
      weights[0][1] = 0.9;
      weights[0][2] = 0.8;
      weights[0][3] = 0.8;
      weights[1][0] = 0.2;
      weights[1][1] = 0.2;
   }
   
   /**
    * Fills the weights array with randomized weights between randomWeightMin and randomWeightMax.
    */
   public void fillRandomWeights()
   {
      for (int i = 0; i < numActivationsA; i++)
      {
         for (int j = 0; j < numActivationsH; j++)
         {
            weights[i][j] = getRandomWeight();
         }
      }
   }

   /**
    * Generates a random weight between randomWeightMin and randomWeightMax.
    * @return a random weight as a double
    */
   public double getRandomWeight()
   {
      return Math.random() * (randomWeightMax - randomWeightMin) + randomWeightMin;
   }

}
