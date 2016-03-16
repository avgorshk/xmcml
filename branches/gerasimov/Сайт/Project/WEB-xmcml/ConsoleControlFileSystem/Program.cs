using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace ConsoleControlFileSystem
{
    class Program
    {
        private static String TASKS_FOLDER;
        private static String RESULTS_FOLDER;
        private static String PROG_PATH;
        private static String TMP_FOLDER;
        private static String TMP_IMAGE_FOLDER;
        private static String TMP_LOGO_FOLDER;
        private static String TMP_SURFACE_FOLDER;
        private static String LOG_FILE_PATH;

        private const int ONE_HOUR = 720; //*5
        private const int ERROR_MAIN_LOOP = -1;

        private const String FILE_PATHS = "D://PATHS.cfg";
        private static List<InfoOfTask> tasks = new List<InfoOfTask>();

        private struct InfoOfTask
        {
            public String id;
            public String state;
        }

        private static List<int> GetID(String state)
        {
            List<int> IDInitialized = new List<int>();
            for (int i = 0; i < tasks.Count; i++)
            {
                if (String.Equals(tasks[i].state, state))
                {
                    IDInitialized.Add(Convert.ToInt32(tasks[i].id));
                }
            }
            IDInitialized.Sort();
            return IDInitialized;
        }

        private static InfoOfTask ParseInfo(String info)
        {
            InfoOfTask task = new InfoOfTask();
            Char[] separator = { '=', '\r', '\n' };
            String[] tags = info.Split(separator, StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < tags.Length; i += 2)
            {
                switch (tags[i])
                {
                    case "ID":
                        {
                            task.id = tags[i + 1];
                            break;
                        }
                    case "STATE":
                        {
                            task.state = tags[i + 1];
                            break;
                        }
                }
            }
            return task;
        }

        private static void UpdateTasks()
        {
            tasks.Clear();
            DirectoryInfo mainDirectory = new DirectoryInfo(TASKS_FOLDER);
            DirectoryInfo[] directories = mainDirectory.GetDirectories();

            for (int i = 0; i < directories.Length; i++)
            {
                StreamReader reader = new StreamReader(directories[i].FullName + "/INFO.TXT");
                String info = reader.ReadToEnd();
                reader.Close();
                tasks.Add(ParseInfo(info));
            }
        }

        private static void AddMessageInLog(string message)
        {
            StreamWriter writer = new StreamWriter(LOG_FILE_PATH, true);
            writer.WriteLine(message);
            writer.Close();
        }

        private static bool CreateMyLogFile()
        {
            try
            {
                StreamWriter writer = new StreamWriter(LOG_FILE_PATH, false);
                writer.Close();
                return true;
                
            }
            catch (Exception)
            {
                return false;
            }
        }

        private static void CreateDirectoryTask(String pathDirectory)
        {
            if (Directory.Exists(pathDirectory))
            {
                Directory.Delete(pathDirectory, true);
                AddMessageInLog("Folder \"" + pathDirectory + "\" is deleted...");
            }
            Directory.CreateDirectory(pathDirectory);
            AddMessageInLog("Folder \"" + pathDirectory + "\" is created...");
        }

        private static void ChangeStateInfoFile(int id, String NEW_STATE)
        {
            if (!File.Exists(TASKS_FOLDER + "/"
                + id.ToString() + "/INFO.TXT"))
                return;

            StreamReader reader = new StreamReader(TASKS_FOLDER + "/"
                + id.ToString() + "/INFO.TXT");

            String info = reader.ReadToEnd();
            reader.Close();

            Char[] separator = { '=', '\r', '\n' };
            String[] tags = info.Split(separator, StringSplitOptions.RemoveEmptyEntries);

            String ID, NAME, DATE_AND_TIME;

            ID = NAME = DATE_AND_TIME = "";

            for (int i = 0; i < tags.Length; i += 2)
            {
                switch (tags[i])
                {
                    case "ID":
                        {
                            ID = tags[i + 1];
                            break;
                        }
                    case "NAME":
                        {
                            NAME = tags[i + 1];
                            break;
                        }
                    case "DATE_AND_TIME":
                        {
                            DATE_AND_TIME = tags[i + 1];
                            break;
                        }
                }
            }
            StreamWriter writer = new StreamWriter(TASKS_FOLDER + "/"
                + id.ToString() + "/INFO.TXT");
            writer.WriteLine("ID=" + ID);
            writer.WriteLine("NAME=" + NAME);
            writer.WriteLine("DATE_AND_TIME=" + DATE_AND_TIME);
            writer.WriteLine("STATE=" + NEW_STATE);
            writer.Close();
        }

        private static void CreateFileOutInfo(int id, System.Diagnostics.Process proc)
        {
            StreamWriter writer = new StreamWriter(RESULTS_FOLDER + "/"
                + id.ToString() + "/OUT.TXT");
            writer.Write(proc.StandardOutput.ReadToEnd());
            writer.Close();
        }

        private static void DeleteFolderTMP()
        {
            try
            {
                if (Directory.Exists(TMP_FOLDER))
                {
                    Directory.Delete(TMP_FOLDER, true);
                }
                Directory.CreateDirectory(TMP_IMAGE_FOLDER);
                Directory.CreateDirectory(TMP_SURFACE_FOLDER);
            }
            catch
            {
                return;
            }
        }

        private static void DeleteFolderLogo()
        {
            try
            {
                if (Directory.Exists(TMP_LOGO_FOLDER))
                {
                    Directory.Delete(TMP_LOGO_FOLDER, true);
                }
                Directory.CreateDirectory(TMP_LOGO_FOLDER);
            }
            catch
            {
                return;
            }
        }

        private static void DeleteFolderTasks(List<int> IDs)
        {
            try
            {
                String inputDir, outputDir;

                for (int i = 0; i < IDs.Count; i++)
                {
                    inputDir = TASKS_FOLDER + "/" + IDs[i].ToString();
                    if (Directory.Exists(inputDir))
                    {
                        Directory.Delete(inputDir, true);
                    }

                    outputDir = RESULTS_FOLDER + "/" + IDs[i].ToString();
                    if (Directory.Exists(outputDir))
                    {
                        Directory.Delete(outputDir, true);
                    }
                }
            }
            catch
            {
                return;
            }
        }

        private static String GetValueFromPaths(String tag)
        {
            if (!File.Exists(FILE_PATHS))
            {
                return null;
            }

            StreamReader reader = new StreamReader(FILE_PATHS);

            int first_index, last_index;
            String line, value = null;

            while (!reader.EndOfStream)
            {
                line = reader.ReadLine();

                first_index = line.IndexOf(tag);

                if (first_index != -1)
                {
                    first_index = line.IndexOf("\"") + 1;
                    last_index = line.LastIndexOf("\"");

                    if ((first_index == -1) || (last_index == -1))
                    {
                        continue;
                    }

                    value = line.Substring(first_index, last_index - first_index);
                    break;
                }
            }
            reader.Close();

            return value;
        }

        private static bool InitMyPaths()
        {
            TASKS_FOLDER = GetValueFromPaths("TASKS_FOLDER");
            RESULTS_FOLDER = GetValueFromPaths("RESULTS_FOLDER");
            PROG_PATH = GetValueFromPaths("PROG_PATH");
            TMP_FOLDER = GetValueFromPaths("TMP_FOLDER");
            TMP_IMAGE_FOLDER = GetValueFromPaths("TMP_IMAGE_FOLDER");
            TMP_LOGO_FOLDER = GetValueFromPaths("TMP_LOGO_FOLDER");
            TMP_SURFACE_FOLDER = GetValueFromPaths("TMP_SURFACE_FOLDER");
            LOG_FILE_PATH = GetValueFromPaths("LOG_FILE_PATH");

            if ((TASKS_FOLDER == null) || (RESULTS_FOLDER == null) || (PROG_PATH == null) ||
                (TMP_FOLDER == null) || (TMP_IMAGE_FOLDER == null) || (TMP_LOGO_FOLDER == null)
                    || (TMP_SURFACE_FOLDER == null) || (LOG_FILE_PATH == null))
            {
                return false;
            }
            return true;
        }

        private static int Running()
        {
            UpdateTasks();
            List<int> IDQueued = GetID("QUEUED");
            DeleteFolderTasks(IDQueued);
            List<int> IDRunning = GetID("RUNNING");
            DeleteFolderTasks(IDRunning);

            Queue<int> queueTasks = new Queue<int>();
            System.Diagnostics.Process proc = null;
            bool isOk = true;
            int id = -1, count = 0, numHours = 0;

            while (isOk)
            {
                UpdateTasks();
                List<int> IDInitialized = GetID("INITIALIZE");

                //Если есть новые задачи добавляем
                if (IDInitialized.Count > 0)
                {
                    for (int i = 0; i < IDInitialized.Count; i++)
                    {
                        queueTasks.Enqueue(IDInitialized[i]);
                        ChangeStateInfoFile(IDInitialized[i], "QUEUED");
                        AddMessageInLog("Task with ID=" + IDInitialized[i].ToString() + " added to the queue...");
                    }
                    AddMessageInLog("");
                }

                //Проверяем закончилось ли выполнение последней задачи
                if ((proc != null) && (proc.HasExited))
                {
                    if (proc.ExitCode == 0)
                    {
                        ChangeStateInfoFile(id, "FINISHED");
                        CreateFileOutInfo(id, proc);
                        AddMessageInLog("Task with ID=" + id.ToString() + " completed successfully...");
                    }
                    else
                    {
                        ChangeStateInfoFile(id, "CRASHED");
                        CreateFileOutInfo(id, proc);
                        AddMessageInLog("Task with ID=" + id.ToString() + " failed with error \""
                            + proc.ExitCode.ToString() + "\"...");
                    }

                    proc.Close();
                    proc = null;
                }

                //Если есть задачи которые можно запустить, то запускаем
                if ((queueTasks.Count > 0) && (proc == null))
                {
                    id = queueTasks.Dequeue();

                    String ID, resultFolderID, arguments;

                    ID = id.ToString();
                    resultFolderID = RESULTS_FOLDER + "/" + ID;
                    arguments = "-i " + TASKS_FOLDER + "/" + ID
                        + "/" + ID + ".xml" + " -s " + TASKS_FOLDER + "/" + ID
                        + "/" + ID + ".surface" + " -o " + resultFolderID
                        + "/" + ID + ".mcml.out";

                    CreateDirectoryTask(resultFolderID);

                    proc = new System.Diagnostics.Process();
                    proc.StartInfo.Arguments = arguments;
                    proc.StartInfo.FileName = PROG_PATH;
                    proc.StartInfo.CreateNoWindow = false;
                    proc.StartInfo.UseShellExecute = false;
                    proc.StartInfo.RedirectStandardOutput = true;
                    proc.Start();

                    ChangeStateInfoFile(id, "RUNNING");
                    AddMessageInLog("Task with ID=" + ID + " loaded with args \"" + arguments + "\"...");
                }

                if (count < ONE_HOUR)//СЧИТАЕМ ЧАС
                {
                    count += 1;
                    System.Threading.Thread.Sleep(5000);
                }
                else//ПРОВЕРЯЕМ НАСТУПИЛО ИЛИ НЕТ ВРЕМЯ ОЧИСТКИ
                {
                    count = 0;

                    DeleteFolderTMP();
                    DeleteFolderLogo();
                    AddMessageInLog("Cleaning of temporary folders completed...");

                    numHours += 1;
                    if (numHours == 48)
                    {
                        List<int> IDFinished = GetID("FINISHED");
                        DeleteFolderTasks(IDFinished);
                        List<int> IDCrashed = GetID("CRASHED");
                        DeleteFolderTasks(IDCrashed);

                        AddMessageInLog("Cleaning of tasks completed...");
                    }
                    if (numHours > 48)
                    {
                        numHours = 0;
                    }
                }
            }
            AddMessageInLog("Unknown error " + ERROR_MAIN_LOOP.ToString() + "!");
            return ERROR_MAIN_LOOP;
        }

        private static void Main(String[] args)
        {
            bool pathsOk = InitMyPaths();
            bool logOk = CreateMyLogFile();

            if (!logOk)
            {
                return;
            }
            else if (!pathsOk)
            {
                AddMessageInLog("Config file with paths \"" + FILE_PATHS + "\" not found!");
                return;
            }
            else
            {
                AddMessageInLog("Start service...");
                int error = Running();
            }
            return;
        }
    }
}