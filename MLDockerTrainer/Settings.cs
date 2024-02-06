namespace MLDockerTrainer
{
    public class Settings
    {
        public string ModelToTrain { get; set; }
        public List<string> DataPath { get; set; }
        public int TrainingSplit { get; set; }
        public int ValidationSplit { get; set; }
        public int TestSplit { get; set; }

        public Settings(string path)
        {
            //Read the settings from the file (txt file)
            using (var reader = new System.IO.StreamReader(path))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var split = line.Split('=');
                    if (split.Length != 2)
                    {
                        throw new Exception("Invalid settings file");
                    }

                    var key = split[0].Trim();
                    var value = split[1].Trim();

                    switch (key)
                    {
                        case "ModelToTrain":
                            ModelToTrain = value;
                            break;
                        case "DataPath":
                            DataPath = value.Split(',').ToList();
                            break;
                        case "TrainingSplit":
                            TrainingSplit = int.Parse(value);
                            break;
                        case "ValidationSplit":
                            ValidationSplit = int.Parse(value);
                            break;
                        case "TestSplit":
                            TestSplit = int.Parse(value);
                            break;
                        default:
                            throw new Exception("Invalid settings file");
                    }
                }
            }
        }
    }
}
