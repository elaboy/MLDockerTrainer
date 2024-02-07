using Proteomics.PSM;
namespace MLDockerTrainer.Utils
{
    public static class PsmParser
    {
        public static List<List<string>> GetRetentionTimeWithFullSequenceAsTokens(List<PsmFromTsv> psmList)
        {
            var tokens = new List<List<string>>();

            foreach (var psm in psmList)
            {
                var retentionTime = psm.RetentionTime;
                var fullSequence = psm.FullSequence;

                List<string> tokenList = new();
                tokenList.Add(TokenKit.RETENTION_TIME_START_TOKEN);

                retentionTime = Math.Round(retentionTime.Value, 2, MidpointRounding.AwayFromZero);

                tokenList.AddRange(RetentionTimeTokenizer(retentionTime.Value));
                tokenList.Add(TokenKit.END_OF_RETENTION_TIME_TOKEN);
                tokenList.Add(TokenKit.START_OF_SEQUENCE_TOKEN);
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
                //}
                tokenList.Add(TokenKit.END_OF_SEQUENCE_TOKEN);

                tokens.Add(tokenList);
            }

            return tokens;
        }

        private static string[] RetentionTimeTokenizer(double retentionTime)
        {
            var tokens = new string[5];
            var retentionTimeAsString = retentionTime.ToString().Split('.');
            var integers = retentionTimeAsString[0];
            var decimals = retentionTimeAsString.Count() == 2 ? retentionTimeAsString[1] : "00"; //if there is no decimal part, add 00
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
    }
}
