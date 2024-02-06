using CommandLine;
using MLDockerTrainer;

public static class Program
{

    public static void Main(string[] args)
    {
        var parser = new Parser(with => with.HelpWriter = null);
        var result = parser.ParseArguments<Settings>(args);

        result.WithParsed(settings => Run(settings));

    }

    public static void Run(Settings settings)
    {
        
    }
}
