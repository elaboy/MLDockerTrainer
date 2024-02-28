using System.Data;
using System.Text;
using System.Text.RegularExpressions;
using Easy.Common.Extensions;
using MathNet.Numerics.Statistics;
using TorchSharp;

namespace MLDockerTrainer.Utils
{
    public class RetentionTimeCalibrator
    {
        public List<string> FullSequences { get; set; }
        public Dictionary<string, Dictionary<string, double>> FileDictionary { get; set; }
        public int FileCount { get; set; }
        public Dictionary<string, (double[], double, double)> RetentionTimeDictionary { get; set; } //FullSequence, (RetentionTime[], mean, variance)

        //todo: Update this to use same methods as the other constructor
        public RetentionTimeCalibrator(Settings settings)
        {
            Dictionary<string, Dictionary<string, double>> fileDictionary = new Dictionary<string, Dictionary<string, double>>();

            int fileCount = settings.DataPath.Count;

            foreach (var file in settings.DataPath)
            {
                Dictionary<string, double> fullSequenceAndRetentionTimeDictionary = new Dictionary<string, double>();

                var psms = Readers.SpectrumMatchTsvReader.ReadPsmTsv(file, out var warnings)
                    .Where(x => x.AmbiguityLevel == "1" &&
                                x.DecoyContamTarget.Equals("T") &&
                                x.QValue < 0.01 && 
                                x.PEP < 0.5);

                psms.ForEach(x => fullSequenceAndRetentionTimeDictionary.Add(x.FullSequence, x.RetentionTime is not null ? x.RetentionTime.Value : 0));

                fileDictionary.Add(file, fullSequenceAndRetentionTimeDictionary);
            }

            //Get the full sequences and add them to the RetentionTimeDictionary. 
            //null values for sequences that are not present in the file
            RetentionTimeDictionary = new Dictionary<string, (double[], double, double)>();
            FullSequences = fileDictionary.SelectMany(x => x.Value.Keys).Distinct().ToList();
            FileDictionary = fileDictionary;
            FileCount = fileCount;

            foreach (var sequence in FullSequences)
            {
                var retentionTimeList = new List<double>();

                foreach (var file in FileDictionary)
                {

                    if (file.Value.ContainsKey(sequence))
                    {
                        retentionTimeList.Add(file.Value[sequence]);
                    }
                }

                RetentionTimeDictionary.Add(sequence, (retentionTimeList.ToArray(), 0, 0));
            }
        }

        public RetentionTimeCalibrator(List<string> paths)
        {
            Dictionary<string, Dictionary<string, double>> fileDictionary = new Dictionary<string, Dictionary<string, double>>();

            int fileCount = paths.Count;

            foreach (var file in paths)
            {
                Dictionary<string, double> fullSequenceAndRetentionTimeDictionary = new Dictionary<string, double>();

                var psms = Readers.SpectrumMatchTsvReader.ReadPsmTsv(file, out var warnings)
                    .Where(x => x.AmbiguityLevel == "1" &&
                           x.DecoyContamTarget.Equals("T") &&
                           x.QValue < 0.01 &&
                           x.PEP < 0.5);

                psms.ForEach(x => 
                    fullSequenceAndRetentionTimeDictionary
                        .Add(x.FullSequence, x.RetentionTime != null ? x.RetentionTime.Value : 0));

                fileDictionary.Add(file, fullSequenceAndRetentionTimeDictionary);
            }

            //Get the full sequences and add them to the RetentionTimeDictionary. 
            //null values for sequences that are not present in the file
            RetentionTimeDictionary = new Dictionary<string, (double[], double, double)>();
            FullSequences = fileDictionary.SelectMany(x => x.Value.Keys).Distinct().ToList();
            FileDictionary = fileDictionary;
            NormalizeRetentionTimes();
            FileCount = fileCount;

            foreach (var sequence in FullSequences)
            {
                var retentionTimeList = new List<double>();

                foreach (var file in FileDictionary)
                {

                    if (file.Value.ContainsKey(sequence))
                    {
                        retentionTimeList.Add(file.Value[sequence]);
                    }
                }

                RetentionTimeDictionary.Add(sequence, (retentionTimeList.ToArray(), retentionTimeList.Average(), retentionTimeList.Variance()));
            }
            //order retention times by the mean
            RetentionTimeDictionary = RetentionTimeDictionary;
            //.OrderBy(x => x.Value.Item2)
            //.ToDictionary(x => x.Key, 
            //    x => x.Value);
        }

        public DataTable GetDataAsDataTable()
        {
            DataTable dataTable = new DataTable();
            dataTable.Columns.Add("FullSequence", typeof(string));
            foreach (var file in FileDictionary.Keys)
            {
                dataTable.Columns.Add(file, typeof(string));
            }
            dataTable.Columns.Add("Mean", typeof(double));
            dataTable.Columns.Add("Variance", typeof(double));

            foreach (var sequence in RetentionTimeDictionary)
            {
                var row = dataTable.NewRow();

                ////regex to replace outer brackets with *, not inner ones
                //string replacedBracketsWithStar = Regex.Replace(sequence.Key,
                //    "(?<=[A-HJ-Z])\\[|(?<=\\[)[A-HJ-Z](?=\\])|(?<=[A-HJ-Z])\\](?=$|[A-Z]|(?<=\\])[^A-Z])",
                //    "*");

                ////regex to replace between * and : inside the mod 
                //string noColonModWithStar = Regex.Replace(replacedBracketsWithStar, "\\*(.*?):",
                //    "*");

                //row["FullSequence"] = noColonModWithStar.ToString();

                row["FullSequence"] = sequence.Key;

                var retentionTimes = sequence.Value.Item1;

                for (int i = 0; i < retentionTimes.Length; i++)
                {
                    row[i + 1] = retentionTimes[i];
                }

                row["Mean"] = sequence.Value.Item2;
                row["Variance"] = sequence.Value.Item3;
                dataTable.Rows.Add(row);
            }

            return dataTable;
        }

        public static void ToCSV(DataTable dataTable, string filePath)
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


        private void NormalizeRetentionTimes()
        {
            foreach (var file in FileDictionary)
            {
                var max = file.Value.Values.Max();
                var min = file.Value.Values.Min();

                foreach (var sequence in file.Value)
                {
                    file.Value[sequence.Key] = (sequence.Value - min) / (max - min);
                }
            }

        }

    }
}
