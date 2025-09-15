/**
 * Main class that runs the A-B-1 neural network based on user configurations loaded from a network configuration file. 
 * It stores and sets the configuration parameters, allocates memory for the Network, trains/runs the 
 * Network, and outputs relevant information from running the Network.
 * 
 * @author Brenna Ren
 * @version September 15, 2025
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
      network.setManualConfigs();
      network.printNetworkConfigs();
      network.allocateNetworkMemory();
      network.populateNetwork();

      if (network.printInputTable)
      {
         network.printInputTable();
      }

      if (network.isTraining)
      {
         network.printTrainingParameters();
         network.train();
         network.printTrainResults();

         if (network.runAfterTraining)
         {
            network.run();
            network.printRunResults();
         }
      } // if (network.isTraining)
      else
      {
         network.run();
         network.printRunResults();
      }

   } // public static void main(String[] args)
} // public class Main