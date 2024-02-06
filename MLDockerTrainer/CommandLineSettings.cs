using CommandLine;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLDockerTrainer
{
    public class CommandLineSettings
    {
        [Option('f', HelpText = "Text file with settings for the training")]
        public Settings Settings { get; set; }

        


    }
}
