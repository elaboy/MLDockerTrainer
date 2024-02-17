namespace Tests
{
    public class TestMainMethod
    {
        [Test]
        public void TestMainMethodRun()
        {
            Program.Main(new string[] { @"C:\Users\elabo\Documents\GitHub\MLDockerTrainer\settings.txt" });
        }
    }
}
