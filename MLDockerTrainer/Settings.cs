namespace MLDockerTrainer
{
    public class Settings
    {
        #region Model Parameters

        public string ModelToTrain { get; set; }

        //if transformer, then: 
        public string? VocabularyFilePath { get; set; }
        public int? SourceVocabularySize { get; set; }
        public int? TargetVocabularySize { get; set; }
        public int? SourceSequenceLength { get; set; }
        public int? TargetSequenceLength { get; set; }
        public int? DModel { get; set; }
        public int? N { get; set; }
        public int? H { get; set; }
        public double? Dropout { get; set; }
        public int? DFF { get; set; }

        #endregion

        #region Data parameters

        public List<string> DataPath { get; set; }
        public int TrainingSplit { get; set; }
        public int ValidationSplit { get; set; }
        public int TestSplit { get; set; }
        
        #endregion

        #region Training Parameters

        public int BatchSize { get; set; }
        public int Epochs { get; set; }
        public double LearningRate { get; set; }
        public bool UseLearningRateScheduler { get; set; }
        public double? LearningRateDecay { get; set; }
        public int? LearningRateDecayStep { get; set; }

        public bool UseEarlyStopping { get; set; }
        public int? EarlyStoppingPatience { get; set; }

        public bool SaveModelEachEpoch { get; set; }
        //todo: expand this to include more parameters
        #endregion

        public Settings(string path)
        {
            //Read the settings from the file (txt file)
            using (var reader = new System.IO.StreamReader(path))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var key = "";
                    var value = "";

                    var split = line.Split('=');

                    if (split.Length == 1)
                    {
                        key = split[0].Trim();
                        value = null;
                    }
                    else if (split.Length == 2)
                    {
                        key = split[0].Trim();
                        value = split[1].Trim();
                    }
                    else
                    {
                        throw new Exception("Invalid settings file");
                    }

                    #region Assign values to the properties
                    switch (key)
                    {
                        case "ModelToTrain":
                            ModelToTrain = value;
                            break;
                        case "VocabularyFilePath":
                            VocabularyFilePath = value;
                            break;
                        case "SourceVocabularySize":
                            SourceVocabularySize = int.Parse(value);
                            break;
                        case "TargetVocabularySize":
                            TargetVocabularySize = int.Parse(value);
                            break;
                        case "SourceSequenceLength":
                            SourceSequenceLength = int.Parse(value);
                            break;
                        case "TargetSequenceLength":
                            TargetSequenceLength = int.Parse(value);
                            break;
                        case "DModel":
                            DModel = int.Parse(value);
                            break;
                        case "N":
                            N = int.Parse(value);
                            break;
                        case "H":
                            H = int.Parse(value);
                            break;
                        case "Dropout":
                            Dropout = double.Parse(value);
                            break;
                        case "DFF":
                            DFF = int.Parse(value);
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
                        case "BatchSize":
                            BatchSize = int.Parse(value);
                            break;
                        case "Epochs":
                            Epochs = int.Parse(value);
                            break;
                        case "LearningRate":
                            LearningRate = double.Parse(value);
                            break;
                        case "UseLearnignRateScheduler":
                            UseLearningRateScheduler = bool.Parse(value);
                            break;
                        case "LearningRateDecay":
                            LearningRateDecay = double.Parse(value);
                            break;
                        case "LearningRateDecayStep":
                            LearningRateDecayStep = int.Parse(value);
                            break;
                        case "UseEarlyStopping":
                            UseEarlyStopping = bool.Parse(value);
                            break;
                        case "EarlyStoppingPatience":
                            EarlyStoppingPatience = int.Parse(value);
                            break;
                        case "SaveModelEachEpoch":
                            SaveModelEachEpoch = bool.Parse(value);
                            break;
                        default:
                            throw new Exception("Invalid settings file");
                    }
                    #endregion
                }
            }
        }
    }
}
