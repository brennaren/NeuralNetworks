import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class ApplyOnesComplement {
    public static void main (String[] args)
    {
        for (int x=1; x<=5; x++)
        {
            for (int y=1; y<=6; y++)
            {
                int biHeight = 256;
                int biWidth = 256;
                
                String modifiedBmpString = "onesBMP/scaledCropped" + x + "-" + y + ".bmp";
                String modifiedBinString = "onesBin/scaledCropped" + x + "-" + y + ".bin";
                
                String origBinString = "finalBin/scaledCropped" + x + "-" + y + ".bin";

                int[][] imageArray = new int[biHeight][biWidth];
                int[][] onesComplimentArray = new int[biHeight][biWidth];
        
                try
                {
                    FileInputStream fstream = new FileInputStream(origBinString);
                    DataInputStream in = new DataInputStream(fstream);

                    for (int i = biHeight - 1; i >= 0; --i)    // write over the rows (in the usual inverted format)
                    {
                        for (int j = 0; j < biWidth; ++j) // and the columns
                        {
                            double activationVal = in.readDouble();
                            byte byteVal = (byte)(activationVal * 255.0);    // $$$
                            imageArray[i][j] = byteVal;
                            // System.out.print(imageArray[i][j] + " ");
                        }
                        // System.out.println();
                    }
                
                    in.close();
                    fstream.close();
                }
                catch (Exception e)
                {
                    System.err.println("File input error" + e);
                }

                PelArray pixels = new PelArray(imageArray);

                pixels = pixels.onesComplimentImage();

                onesComplimentArray = new int[biHeight][biWidth];
                onesComplimentArray = pixels.getPelArray();

                try
                {
                    FileOutputStream fstream = new FileOutputStream(modifiedBinString);
                    DataOutputStream out = new DataOutputStream(fstream);

                    for (int i = biHeight - 1; i >= 0; --i)    // write over the rows (in the usual inverted format)
                    {
                        for (int j = 0; j < biWidth; ++j) // and the columns
                        {
                            int pel = onesComplimentArray[i][j];
                            double activationVal  = (byte)(pel & 0x00FF) / 255.0;

                            out.writeDouble(activationVal);
                        }
                    }
                
                    out.close();
                    fstream.close();
                }
                catch (Exception e)
                {
                    System.err.println("File output error" + e);
                }

                // check that BGR2RMP is taking in 0-256 activation values and not 0-1
                BGR2BMP.main(new String[] {"gray", ""+biWidth, ""+biHeight, modifiedBinString, modifiedBmpString});

                // check that BMP2OneByte is outputting doubles and BGR2BMP is taking in 0-1 activation values and not 0-256
                // BMP2OneByte.main(new String[] {croppedBmpString, modifiedBinString});
                // BGR2BMP.main(new String[] {"gray", ""+125, ""+110, modifiedBinString, modifiedBmpString});
            }
        }
    }
}
