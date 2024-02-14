using CsvHelper;
using Easy.Common.Extensions;
using Proteomics.PSM;
using System.Globalization;
using System.Text.RegularExpressions;
using TorchSharp;

namespace MLDockerTrainer.Utils;

public static class TokenKit
{
    //Current constant comments are refering to the VocabularyForTransformerUniprot_V2.csv file Token ids
    public const string PADDING_TOKEN = "<PAD>"; //0
    public const string RETENTION_TIME_START_TOKEN = "<RT>"; //1
    public const string END_OF_RETENTION_TIME_TOKEN = "</RT>"; //2
    public const string END_OF_SEQUENCE_TOKEN = "<EOS>"; //3
    public const string START_OF_SEQUENCE_TOKEN = "<SOS>"; //4
    public const string MASKING_TOKEN = "<MASK>"; //5

    public static List<string> TokenizeRetentionTimeWithFullSequence(PsmFromTsv psm)
    {
        var retentionTime = psm.RetentionTime;
        var fullSequence = psm.FullSequence;

        List<string> tokenList = new();
        tokenList.Add(RETENTION_TIME_START_TOKEN);

        retentionTime = Math.Round(retentionTime.Value, 2, MidpointRounding.AwayFromZero);

        tokenList.AddRange(RetentionTimeTokenizer(retentionTime.Value));
        tokenList.Add(END_OF_RETENTION_TIME_TOKEN);
        tokenList.Add(START_OF_SEQUENCE_TOKEN);
        var fullSequenceSplit = fullSequence.Split('[', ']');
        foreach (var item in fullSequenceSplit)
        {
            if (!item.Contains(" "))
            {
                foreach (var residue in item)
                {
                    tokenList.Add(residue.ToString());
                }
            }
            else
            {
                var splitByColon = item.Split(':');
                tokenList.Add(splitByColon[1]);
            }
        }
        //tokenList.Add(fullSequence);

        //// Pad the rest of the tokens
        //var paddingRequired = (tokenLength - tokenList.Count) - 1;

        //for (int i = 0; i < paddingRequired; i++)
        //{
        //    tokenList.Add(PADDING_TOKEN);
        //}
        tokenList.Add(END_OF_SEQUENCE_TOKEN);

        return tokenList;
    }

    public static List<List<string>> TokenizeRetentionTimeWithFullSequence(CalibratedRetentionTimes calibratedRTs)
    {
        var dictionary = calibratedRTs.RetentionTimeDictionary;

        var tokens = new List<List<string>>();
        //todo: get rid of the regex expressions not being used and save them for later with a description

        foreach (var sequence in dictionary)
        {
            List<string> tokenList = new();
            tokenList.Add(RETENTION_TIME_START_TOKEN);

            //regex to match the mods
            MatchCollection matches = Regex.Matches(sequence.Key, "\\w+:(\\w+(?:\\[[^\\[\\]]+\\])? on \\w)"); 

            //regex to match outer brackets with *, not inner ones
            MatchCollection matchesForReplacement = Regex.Matches(sequence.Key,
                "(?<=[A-HJ-Z])\\[|(?<=\\[)[A-HJ-Z](?=\\])|(?<=[A-HJ-Z])\\](?=$|[A-Z]|(?<=\\])[^A-Z])");

            //regex to replace outer brackets with *, not inner ones
            string replacedBracketsWithStar = Regex.Replace(sequence.Key,
                "(?<=[A-HJ-Z])\\[|(?<=\\[)[A-HJ-Z](?=\\])|(?<=[A-HJ-Z])\\](?=$|[A-Z]|(?<=\\])[^A-Z])", 
                "*");

            //regex to replace between * and : inside the mod 
            string noColonModWithStar = Regex.Replace(replacedBracketsWithStar, "\\*(.*?):",
                "*");

            //regex to match between * and : inside the mod
            MatchCollection matchesForNoColonModWithStar = Regex.Matches(noColonModWithStar, "\\*(.*?)\\*");

            var retentionTime = Math.Round(sequence.Value.Value, 2, MidpointRounding.AwayFromZero);

            tokenList.AddRange(RetentionTimeTokenizer(retentionTime));
            tokenList.Add(END_OF_RETENTION_TIME_TOKEN);
            tokenList.Add(START_OF_SEQUENCE_TOKEN);

            //splitting by *
            foreach (var segment in replacedBracketsWithStar.Split('*'))
            {
                if (segment.Contains(" ")) //if it contains a split, it is a mod
                {
                    //regex to match between * and : inside the mod
                    var mod = Regex.Replace(segment, "(.*?):", ""); //remove everything before the colon
                    tokenList.Add(mod); //add the mod
                    continue;
                }
                
                foreach (var residue in segment)
                {
                    tokenList.Add(residue.ToString()); //add the residue
                }
            }
            tokenList.Add(END_OF_SEQUENCE_TOKEN);
            tokens.Add(tokenList);
        }
        return tokens;
    }

    public static string[] NumericalTokenizer(double number)
    {
        var numberAsString = number.ToString();
        var integerAndDecimalsSplit = numberAsString.Split('.');

        var integerPart = integerAndDecimalsSplit[0];
        var integerPartLength = integerPart.Length;

        var decimalPart = integerAndDecimalsSplit[1];
        var decimalPartLength = decimalPart.Length;

        string[] tokens = new string[integerPartLength + decimalPartLength];

        var positiveToZeroCounter = integerPartLength - 1;

        for (int i = 0; i < integerPartLength; i++)
        {
            // Add a period at the end of the integer part
            if (positiveToZeroCounter == 0)
            {
                tokens[i] = integerPart[i].ToString() + "_" +
                            (positiveToZeroCounter--).ToString() + ".";
                break;
            }

            tokens[i] = integerPart[i].ToString() + "_" + (positiveToZeroCounter--).ToString();
        }

        var negativeCounter = -1;

        for (int i = 0; i < decimalPartLength; i++)
        {
            tokens[integerPartLength + i] = decimalPart[i].ToString() + "_" + (negativeCounter--).ToString();
        }

        return tokens;
    }

    public static string[] RetentionTimeTokenizer(double retentionTime)
    {
        var tokens = new string[5];
        var retentionTimeAsString = retentionTime.ToString().Split('.');
        var integers = retentionTimeAsString[0];
        var decimals =
            retentionTimeAsString.Count() == 2 ? retentionTimeAsString[1] : "00"; //if there is no decimal part, add 00
        if (integers.Length == 2)
        {
            tokens[0] = 0.ToString();
            tokens[1] = integers[0].ToString();
            tokens[2] = integers[1].ToString();
        }
        else if (integers.Length == 1)
        {
            tokens[0] = 0.ToString();
            tokens[1] = 0.ToString();
            tokens[2] = integers[0].ToString();
        }
        else
        {
            tokens[0] = integers[0].ToString();
            tokens[1] = integers[1].ToString();
            tokens[2] = integers[2].ToString();
        }

        if (decimals.Length == 1)
        {
            tokens[3] = '-' + decimals[0].ToString();
            tokens[4] = 0.ToString();
        }
        else
        {
            tokens[3] = '-' + decimals[0].ToString();
            tokens[4] = '-' + decimals[1].ToString();
        }

        return tokens;
    }

    public static torch.Tensor PaddingTensor(torch.Tensor tensor, int desiredTensorLength)
    {
        if (tensor.shape[0] != desiredTensorLength)
        {
            var padsToAdd = desiredTensorLength - tensor.shape[0];
            var paddingTensor = torch.zeros(padsToAdd);

            var paddedTensor = torch.concat(new List<torch.Tensor>() { tensor, paddingTensor })
                .to_type(torch.ScalarType.Int32);

            paddedTensor[1, -1] = paddedTensor[1, desiredTensorLength - padsToAdd];
        }

        return torch.zeros(1);

    }

    public static List<int> PaddingIntegerList(List<int> integerList, int paddingInteger, int desiredListLength)
    {
        if (integerList.Count < desiredListLength)
        {
            integerList.RemoveAt(integerList.Count - 1); //remove end of sequence token

            var padsToAdd =
                (desiredListLength - integerList.Count) - 1; //the -1 is to leave space for the end of sequence token

            for (int i = 0; i < padsToAdd; i++)
            {
                integerList.Add(paddingInteger);
            }

            integerList.Add(3); //end of sequence token id
        }

        return integerList;
    }

    public static (List<List<int>>, List<List<int>>, List<List<int>>) TrainValidateTestSplit(
        List<List<int>> listOfTokenId, double trainingFraction = 0.8,
        double testingFraction = 0.1, double validationFraction = 0.1)
    {
        var randomizedData = listOfTokenId.Randomize().ToList();

        var trainingSet = randomizedData
            .Take((int)(randomizedData.Count * (1 - validationFraction)))
            .ToList();

        var validationSet = randomizedData
            .Skip((int)(randomizedData.Count * (1 - validationFraction)))
            .Take((int)(randomizedData.Count * (1 - testingFraction)))
            .ToList();

        var testSet = randomizedData
            .Skip((int)(randomizedData.Count * (1 - validationFraction)))
            .ToList();

        return (trainingSet, validationSet, testSet);
    }

    public static List<Token> GetVocab(string vocabPath)
    {
        List<Token> vocab = new();
        using (var reader = new StreamReader(Path.Combine(vocabPath)))
        using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
        {
            vocab.AddRange(csv.GetRecords<Token>().ToList());
        }

        return vocab;
    }

    public static List<List<int>> Tokenize(List<List<string>> psms, List<Token> vocab)
    {
        List<List<int>> tokenizedPsms = new();

        foreach (var tokenList in psms)
        {
            List<int> tokenIdList = new();

            foreach (var token in tokenList)
            {
                if (vocab.Find(x => x.TokenWord == token) is null &&
                    !token.Contains(
                        '_')) //Checks if token is a number, if not clear list and break without adding to main list
                {
                    tokenIdList.Clear();
                    break;
                }

                if (int.TryParse(token[0].ToString(),
                        out var result)) //Takes care of retention time numbers and array positions
                {
                    foreach (var subString in token)
                        tokenIdList.Add(vocab.Find(x => x.TokenWord == subString.ToString()).Id);
                }
                else
                {
                    if (vocab.Any(x =>
                            x.TokenWord ==
                            token)) //takes all the other non numerical tokens and adds their id to the list
                    {
                        tokenIdList.Add(vocab.Find(x => x.TokenWord == token).Id);
                    }
                }
            }

            if (tokenIdList.Count != 0) //Empty list is not added to main list
                tokenIdList =
                    PaddingIntegerList(tokenIdList, 0,
                        100); //makes sure padding is done right with desired length
            else
                continue;

            tokenizedPsms.Add(tokenIdList);
        }

        return tokenizedPsms;
    }
}