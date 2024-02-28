using System.Data;
using System.Text;

namespace MLDockerTrainer.Utils;

public class RetentionFile
{
    public string FileName { get; private set; }
    public string[] FullSequences { get; private set; }
    public double[] RetentionTimes { get; private set; }
    public double[] NormalizedRetentionTimes { get; private set; }

    public RetentionFile(string psmFilePath)
    {
        var psms = Readers.SpectrumMatchTsvReader.ReadPsmTsv(psmFilePath,
                out var warnings)
            .Where(x => x.AmbiguityLevel == "1" && 
                        x.DecoyContamTarget.Equals("T") &&
                        x.QValue < 0.01 &&
                        x.PEP < 0.5)
            .OrderBy(x => x.RetentionTime);
        FileName = psms.First().FileNameWithoutExtension;
        FullSequences = psms.Select(x => x.FullSequence).ToArray();
        RetentionTimes = psms.Select(x => x.RetentionTime is not null ? x.RetentionTime.Value : 0).ToArray();
        NormalizedRetentionTimes = Normalize(RetentionTimes);
    }

    public DataTable GetAsDataTable()
    {
        DataTable dataTable = new DataTable();
        dataTable.Columns.Add("FullSequence", typeof(string));
        dataTable.Columns.Add("RetentionTime", typeof(double));
        dataTable.Columns.Add("NormalizedRetentionTime", typeof(double));
        for (int i = 0; i < FullSequences.Length; i++)
        {
            var row = dataTable.NewRow();
            row["FullSequence"] = FullSequences[i];
            row["RetentionTime"] = RetentionTimes[i];
            row["NormalizedRetentionTime"] = NormalizedRetentionTimes[i];
            dataTable.Rows.Add(row);
        }
        return dataTable;
    }

    public static void SaveAsCSV(DataTable dataTable, string filePath)
    {
        StringBuilder sb = new StringBuilder();

        IEnumerable<string> columnNames = dataTable.Columns.Cast<DataColumn>().
            Select(column => column.ColumnName);
        sb.AppendLine(string.Join(",", columnNames));

        foreach (DataRow row in dataTable.Rows)
        {
            IEnumerable<string> fields = row.ItemArray.Select(field => field.ToString());
            sb.AppendLine(string.Join(",", fields));
        }

        File.WriteAllText(filePath, sb.ToString());
    }

    private double[] Normalize(double[] retentionTimes)
    {
        double max = retentionTimes.Max();
        double min = retentionTimes.Min();
        double[] normalizedRetentionTimes = new double[retentionTimes.Length];
        for (int i = 0; i < retentionTimes.Length; i++)
        {
            normalizedRetentionTimes[i] = (retentionTimes[i] - min) / (max - min);
        }
        return normalizedRetentionTimes;
    }
}
