using MLDockerTrainer.Utils;

namespace Tests.TestUtils
{
    public class TestRetentionFile
    {
        [Test]
        public void TestGetAsDataTable()
        {
            var retentionFile = new RetentionFile(@"C:\Users\elabo\Documents\MannPeptideResults\HEK293_AllPeptides.psmtsv");
            var dataTable = retentionFile.GetAsDataTable();
            RetentionFile.SaveAsCSV(dataTable, @"C:\Users\elabo\Documents\MannPeptideResults\TestingRetentionFileGetAsDataTable.csv");
        }

        [Test]
        public void TestSaveAsCSVBulk()
        {
            List<string> paths = new();
            foreach (var file in Directory.GetFiles(@"C:\Users\elabo\Documents\MannPeptideResults"))
            {
                if (file.Contains("AllPeptides") && !file.Contains(".txt"))
                {
                    var retentionFile = new RetentionFile(file);
                    var dataTable = retentionFile.GetAsDataTable();
                    RetentionFile.SaveAsCSV(dataTable, Path.Combine(@"C:\Users\elabo\Documents\RetentionFileDatasets", retentionFile.FileName+".csv"));
                }
            }


        }
    }
}
