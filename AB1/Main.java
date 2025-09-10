import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Properties;
import java.util.Scanner;

/**
 * Main class that runs the A-B-1 neural network based on user configurations loaded from a network configuration file. 
 * It stores and sets the configuration parameters, allocates memory for the Network, trains/runs the 
 * Network, and outputs relevant information from running the Network.
 * 
 * @author Brenna Ren
 * @version September 9, 2025
 * Date of creation: September 5, 2025
 * 
 * Functions in this class:
 * 
 * 
 * 
 */
public class Main 
{

   /**
    * Prompts the user to enter the path to the configuration file and returns it as a String.
    * 
    * @param scanner the Scanner object used to read user input
    * @return the path to the configuration file as a String
    */
   public static String getConfigFile(Scanner scanner)
   {
      System.out.print("Enter the path to the configuration file: ");
      return scanner.nextLine();
   }

   /**
    * Loads the configuration parameters from the specified configuration file and sets them in the Network object.
    * @param configFile
    * @param network
    */
   public static void loadConfigs(String configFile, Network network)
   {
      Properties prop = new Properties();
      InputStream input = null;

      try 
      {
         input = new FileInputStream(configFile);
         prop.load(input);

        
      } 
      catch (Exception e) 
      {
         e.printStackTrace();
      } 
      finally 
      {
         if (input != null) 
         {
            try 
            {
               input.close();
            } 
            catch (Exception e) 
            {
               e.printStackTrace();
            }
         }
      }
   }


   /**
    * Main method that contains independent methods that run the Network based on the user's configurations.
    * First, it sets the configuration parameters and echos them by printing them to the console. Then, it 
    * allocates memory for the Network and trains/runs it (based on the user's configurations). Afterwards,
    * it will output the run results, which includes information such as training exit information, iterations
    * reached, and average error reached (user-selectable options).  
    * 
    * @param args command line arguments (not used)
    */
   public static void main(String[] args) 
   {
      Network network = new Network();
      network.setManualConfigs();
      network.printTrainingParameters();
      network.allocateNetworkMemory();
      network.populateNetwork();
      // network.printNetworkWeights();

      // if (network.isTraining)
      // {
      //    network.trainAndReport();
      // }
      // else
      // {
      //    network.runAndReport();
      // }

   } // public static void main(String[] args)
} // public class Main