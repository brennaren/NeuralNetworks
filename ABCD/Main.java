/**
 * Main class that runs the A-B-C neural network based on user configurations that are configured from the
 * configurations file specified in the command line argument or the default configurations file. 
 * Weights can be loaded from a file and saved to a file. Test cases are also read from external files.
 * This class makes the Network set the configuration parameters, allocates memory for training/running, trains/runs 
 * the  Network, and outputs relevant information from running the Network.
 * 
 * @author Brenna Ren
 * @version October 29, 2025
 * Date of creation: September 5, 2025
 */
public class Main 
{
/**
 * Main method that contains independent methods that run the Network based on the user's configurations.
 * First, it sets the configuration parameters and echos them by printing them to the console. Then, it 
 * allocates memory for the Network and trains/runs it (based on the user's configurations). Afterwards,
 * it will output the training/running results, which includes all relevant information such as training 
 * exit information, iterations reached, average error reached, and other user-selectable options (truth 
 * table, network weights).
 * @param args command line arguments (not used)
 */
   public static void main(String[] args) 
   {
      Network network = new Network();

      network.initializeVariables();

      if (args.length > 0)
      {
         network.configFilePath = args[0];
      }
      else
      {
         network.configFilePath = Network.DEFAULT_CONFIG_FILE_PATH;
      }

      network.loadConfigsFromFile();
      network.printNetworkConfigs();

      if (network.isTraining)
      {
         network.printTrainingParameters();
      }

      network.allocateNetworkMemory();
      network.populateNetwork();

      if (network.printInputTable)
      {
         network.printInputTable();
      }

      if (network.isTraining)
      {
         network.startTimer();
         network.trainAll();
         network.endTimer();
         network.printTrainResults();

         if (network.saveWeightsToFile)
         {
            network.saveWeightsToFile();
         }

         if (network.runAfterTraining)
         {
            network.runAll();
            network.printRunResults();
         }
      } // if (network.isTraining)
      else
      {
         network.runAll();
         network.printRunResults();
      }
   } // public static void main(String[] args)
} // public class Main