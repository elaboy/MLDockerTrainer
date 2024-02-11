using CsvHelper;
using CsvHelper.Configuration;
using CsvHelper.Configuration.Attributes;
using System.Globalization;

namespace MLDockerTrainer.Utils;
public class CalibratedRetentionTimes
{
    [Name("FullSequence")]
    public List<string> FullSequence { get; private set; }
    [Name("Mean")]
    public List<double?> Mean { get; private set; }
    [Name("Variance")]
    public List<double?> Variance { get; private set; }

    public Dictionary<string, double?> RetentionTimeDictionary { get; set; }
    public CalibratedRetentionTimes(string path)
    {
        //read csv columns FullSequence, Mean, and  Variance
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            //HeaderValidated = null
        };
        using (var reader = new StreamReader(path))
        using (var csv = new CsvReader(reader, config))
        {
            var records = csv.GetRecords<dynamic>().ToList();
            var fullSequence = records.Select(x => x.FullSequence as string).ToList();
            var mean = records.Select(x => double.Parse(x.Mean) as double?).ToList();
            var variance = records.Select(x => double.Parse(x.Variance) as double?).ToList();

            FullSequence = fullSequence;
            Mean = mean;
            Variance = variance;
        }
        RetentionTimeDictionary = new Dictionary<string, double?>();
        for (int i = 0; i < FullSequence.Count; i++)
        {
            RetentionTimeDictionary.Add(FullSequence[i], Mean[i]);
        }
    }
}