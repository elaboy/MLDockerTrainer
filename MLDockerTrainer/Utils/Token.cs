using CsvHelper.Configuration.Attributes;

namespace MLDockerTrainer.Utils;

public class Token
{
    [Name("Id")]
    public int Id { get; set; }
    [Name("Token")]
    public string TokenWord { get; set; }
}