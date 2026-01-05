import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class MakeChanges {
    
    public static void main(String[] args) {
        // BGR2BMP.main(new String[] {"gray", ""+83, ""+120, "5A.bin", "5A.bmp"});
        
        for (int x=1; x<=5; x++)
        {
            for (int y=1; y<=6; y++)
            {
                int biHeight = 3088;
                int biWidth = 2316;

                String bmpString = "rawBMP/" + x + "-" + y + ".bmp";
                String binString = "rawBin/" + x + "-" + y + ".bin";

                String croppedBmpString = "cropped/cropped" + x + "-" + y + ".bmp";
                String croppedBinString = "cropped/cropped" + x + "-" + y + ".bin";
                
                String modifiedBmpString = "finalBMP/scaledCropped" + x + "-" + y + ".bmp";
                String modifiedBinString = "finalBin/scaledCropped" + x + "-" + y + ".bin";

                int[][] imageArray = new int[biHeight][biWidth];
                int[][] croppedImageArray;

                // BMP2OneByte.main(new String[] {bmpString, binString});
        
                try
                {
                    FileInputStream fstream = new FileInputStream(binString);
                    DataInputStream in = new DataInputStream(fstream);

                    for (int i = biHeight - 1; i >= 0; --i)    // write over the rows (in the usual inverted format)
                    {
                        for (int j = 0; j < biWidth; ++j) // and the columns
                        {
                            imageArray[i][j] = in.readByte();
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

                System.out.printf("Image Width %d\n", pixels.getWidth());
                System.out.printf("Image Height %d\n\n", pixels.getHeight());

                // rough crop
                PelArray roughCroppedPixels = pixels.crop(150, 250, 2249, 2999);

                System.out.printf("Rough Crop Image Width %d\n", roughCroppedPixels.getWidth());
                System.out.printf("Rough Crop Image Height %d\n\n", roughCroppedPixels.getHeight());

                // scale
                roughCroppedPixels = roughCroppedPixels.scale(150, 196);

                roughCroppedPixels = roughCroppedPixels.onesComplimentImage();
                // roughCroppedPixels.dump();

                roughCroppedPixels = roughCroppedPixels.forceMin(75, 0);

                roughCroppedPixels = roughCroppedPixels.removeBackgroundNoise(PelArray.BLACK, 0, 5, 0);

                int xCom = roughCroppedPixels.getXcom();
                int yCom = roughCroppedPixels.getYcom();

                System.out.printf("xCom %d\n", xCom);
                System.out.printf("yCom %d\n\n", yCom);
        
                PelArray croppedPixels = roughCroppedPixels.crop(xCom-60, yCom-92, xCom+64, yCom+17);

                croppedPixels = croppedPixels.removeBackgroundNoise(PelArray.BLACK, 10, 5, 23);
                // croppedPixels = croppedPixels.removeBackgroundNoise(PelArray.BLACK, 15, 5, 5);
                croppedPixels = croppedPixels.removeBackgroundNoiseCorners(PelArray.BLACK, 30, 5, 25);
                croppedPixels = croppedPixels.removeBackgroundNoiseCorners(PelArray.BLACK, 0, 5, 55);
                croppedPixels = croppedPixels.removeBackgroundNoiseCorners(PelArray.BLACK, 0, 5, 55);
                croppedPixels = croppedPixels.removeBackgroundNoise(PelArray.BLACK, 0,6, 0);
                croppedPixels = croppedPixels.removeBackgroundNoise(PelArray.BLACK, 0,6, 0);
                
                int croppedBiWidth = croppedPixels.getWidth();
                int croppedBiHeight = croppedPixels.getHeight();

                System.out.printf("Cropped Image Width %d\n", croppedBiWidth);
                System.out.printf("Cropped Image Height %d\n\n", croppedBiHeight);

                croppedPixels = croppedPixels.onesComplimentImage();

                croppedImageArray = new int[croppedBiHeight][croppedBiWidth];
                croppedImageArray = croppedPixels.getPelArray();

                try
                {
                    FileOutputStream fstream = new FileOutputStream(croppedBinString);
                    DataOutputStream out = new DataOutputStream(fstream);

                    for (int i = croppedBiHeight - 1; i >= 0; --i)    // write over the rows (in the usual inverted format)
                    {
                        for (int j = 0; j < croppedBiWidth; ++j) // and the columns
                        {
                            int pel = croppedImageArray[i][j];
                            byte byteVal  = (byte)(pel & 0x00FF);

                            out.writeByte(byteVal);
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
                BGR2BMP.main(new String[] {"gray", ""+croppedBiWidth, ""+croppedBiHeight, croppedBinString, croppedBmpString});

                // check that BMP2OneByte is outputting doubles and BGR2BMP is taking in 0-1 activation values and not 0-256
                BMP2OneByte.main(new String[] {croppedBmpString, modifiedBinString});
                BGR2BMP.main(new String[] {"gray", ""+125, ""+110, modifiedBinString, modifiedBmpString});
            }
        }

    }
}
