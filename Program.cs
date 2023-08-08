using Datasets;
using MatthiWare.CommandLine;
using Mathematics;
using Models;
using MatthiWare.CommandLine.Core.Models;
using ImageMagick;

public class Program
{
    public static float[,] ConvertImageToFloatArray(string imagePath)
    {
        // Read the image file using ImageMagick
        using (MagickImage image = new MagickImage(imagePath))
        {
            // Resize the image if needed (optional).
            // image.Resize(new MagickGeometry(Width, Height));

            // Get the pixel data as byte array.
            var pixelData = image.GetPixels().GetValues();
            
            // Create a 2D float array with the same dimensions as the image.
            float[,] floatArray = new float[image.Height, image.Width];

            // Iterate over each pixel in the image.
            int bytePerPixel = pixelData.Length / (image.Width * image.Height);
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    // Calculate the position in the byte array.
                    int position = (y * image.Width + x) * bytePerPixel;

                    // Get the grayscale value from the byte array.
                    float grayscaleValue = (float)(pixelData[position] / 65535.0);

                    // Assign the grayscale value to the corresponding position in the float array.
                    floatArray[y, x] = grayscaleValue;
                }
            }

            // Return the resulting float array.
            return floatArray;
        }
    }

    static void Main(string[] args)
    {
        // Command line parser options:
        // "--" before parameter name,
        // "=" after parameter name.
        var options = new CommandLineParserOptions
        {
            PrefixLongOption = "--",
            PostfixOption = "=",
        };

        var parser = new CommandLineParser<Options.ModelOptions>(options);
        var argumentParsing = parser.Parse(args);

        if (argumentParsing.HasErrors)
        {
            Console.Error.WriteLine("Wrong parsing!");
            return;
        }

        // Parsed args.
        var arguments = argumentParsing.Result;

        var dataset = new Mnist();
        var model = new MnistModel(arguments);

        for (int epoch = 0; epoch < arguments.Epochs; epoch++)
        {
            model.TrainEpoch(dataset.Train);
            float[] accCorrects = model.Evaluate(dataset.Test);
            Console.WriteLine($"Epoch: {epoch + 1}, Accuracy: {accCorrects[0]}, Corrects: {accCorrects[1]}");
        }
        
        for (int num = 0; num < 10; num++)
        {
            // Directory of the images of the number.
            string root = $"MNIST-JPG-testing/{num}";
            
            // All images of the particular number.
            var files = from file in Directory.EnumerateFiles(root) select file;
            
            // Value for accuracy calculation.
            float count = 0;
            float correct = 0;
            foreach (var file in files)
            {
                count += 1;
                // Image to 2D float array.
                var image = ConvertImageToFloatArray(file);
                
                // Predict the sample
                float pred = model.PredictSample(image);
                if ((int)pred == num) correct += 1;
            }
            
            // Calculate the accuracy.
            float accuracy = count == 0 ? 0 : correct / count;
            Console.WriteLine($"Number {num}: {(int)(accuracy * 100)}%");
        }
    }
}

