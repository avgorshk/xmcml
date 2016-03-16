using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;

namespace WebServiceControlTasks
{
    /// <summary>
    /// Сводное описание для Service1
    /// </summary>
    [WebService(Namespace = "WEB-xmcml")]
    [WebServiceBinding(ConformsTo = WsiProfiles.BasicProfile1_1)]
    [System.ComponentModel.ToolboxItem(false)]
    // Чтобы разрешить вызывать веб-службу из скрипта с помощью ASP.NET AJAX, раскомментируйте следующую строку. 
    // [System.Web.Script.Services.ScriptService]
    public class Service1 : System.Web.Services.WebService
    {
        private const String TASKS = "TASKS";
        private const String TASKS_FOLDER = "TASKS_FOLDER";
        private const String RESULTS_FOLDER = "RESULTS_FOLDER";

        private const String FILE_PATHS = "D://PATHS.cfg";

        private struct Task
        {
            public String id;
            public String name;
            public String dateAndTime;
            public String state;
        }

        private String GetValueFromPaths(String tag)
        {
            if (!System.IO.File.Exists(FILE_PATHS))
            {
                return null;
            }

            System.IO.StreamReader reader = new System.IO.StreamReader(FILE_PATHS);

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

        private Task ParseInfo(String info)
        {
            Task task = new Task();
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
                    case "NAME":
                        {
                            task.name = tags[i + 1];
                            break;
                        }
                    case "DATE_AND_TIME":
                        {
                            task.dateAndTime = tags[i + 1];
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

        [WebMethod]
        public void Init()
        {
            String pathTasksFolder;
            Application[TASKS_FOLDER] = pathTasksFolder = GetValueFromPaths("TASKS_FOLDER");
            Application[RESULTS_FOLDER] = GetValueFromPaths("RESULTS_FOLDER");

            System.IO.DirectoryInfo mainDirectory = new System.IO.DirectoryInfo(pathTasksFolder);
            System.IO.DirectoryInfo[] directories = mainDirectory.GetDirectories();

            List<Task> tasks = new List<Task>();

            for (int i = 0; i < directories.Length; i++)
            {
                System.IO.StreamReader reader = new System.IO.StreamReader(directories[i].FullName + "/INFO.TXT");
                String info = reader.ReadToEnd();
                reader.Close();

                tasks.Add(ParseInfo(info));
            }

            Application[TASKS] = tasks;
        }

        [WebMethod]
        public int GetNumTasks()
        {
            return ((List<Task>)Application[TASKS]).Count;
        }

        private int CompareTaskFieldID(Task task1, Task task2)
        {
            int id1, id2;

            id1 = Convert.ToInt32(task1.id);
            id2 = Convert.ToInt32(task2.id);

            return id1.CompareTo(id2);
        }

        private int CompareTaskFieldName(Task task1, Task task2)
        {
            return task1.name.CompareTo(task2.name);
        }

        private int CompareTaskFieldDaT(Task task1, Task task2)
        {
            DateTime dat1, dat2;

            DateTime.TryParse(task1.dateAndTime, out dat1);
            DateTime.TryParse(task2.dateAndTime, out dat2);

            return dat1.CompareTo(dat2);
        }

        private int CompareTaskFieldState(Task task1, Task task2)
        {
            return task1.state.CompareTo(task2.state);
        }

        [WebMethod]
        public String[] GetTableStr(int columnSort)
        {
            List<Task> tasks = (List<Task>)Application[TASKS];

            switch (columnSort)
            {
                case 0:
                    {
                        tasks.Sort(CompareTaskFieldID);
                        break;
                    }
                case 1:
                    {
                        tasks.Sort(CompareTaskFieldName);
                        break;
                    }
                case 2:
                    {
                        tasks.Sort(CompareTaskFieldDaT);
                        break;
                    }
                case 3:
                    {
                        tasks.Sort(CompareTaskFieldState);
                        break;
                    }
            }

            const int numColumns = 4;
            String[] strTasks = new String[tasks.Count * numColumns];

            for (int i = 0; i < tasks.Count; i++)
            {
                strTasks[i * numColumns] = tasks[i].id;
                strTasks[i * numColumns + 1] = tasks[i].name;
                strTasks[i * numColumns + 2] = tasks[i].dateAndTime;

                switch (tasks[i].state)
                {
                    case "INITIALIZE":
                        {
                            strTasks[i * numColumns + 3] = "Инициализация";
                            break;
                        }
                    case "QUEUED":
                        {
                            strTasks[i * numColumns + 3] = "В очереди";
                            break;
                        }
                    case "RUNNING":
                        {
                            strTasks[i * numColumns + 3] = "Выполняется";
                            break;
                        }
                    case "FINISHED":
                        {
                            strTasks[i * numColumns + 3] = "Завершено";
                            break;
                        }
                    case "CRASHED":
                        {
                            strTasks[i * numColumns + 3] = "Ошибка";
                            break;
                        }
                }
            }

            return strTasks;
        }

        [WebMethod]
        public List<int> GetID(String state)
        {
            List<Task> tasks = (List<Task>)Application[TASKS];

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

        [WebMethod]
        public String GetPathDownloadOUT(int id)
        {
            String path = (String)Application[RESULTS_FOLDER] + "/" + id.ToString() + "/OUT.TXT";

            if (System.IO.File.Exists(path))
            {
                return path;
            }
            else
            {
                return null;
            }
        }

        [WebMethod]
        public String GetPathDownloadMCML(int id)
        {
            String path = (String)Application[RESULTS_FOLDER] + "/" + id.ToString() + "/" + id.ToString() + ".mcml.out";

            if (System.IO.File.Exists(path))
            {
                return path;
            }
            else
            {
                return null;
            }
        }

        [WebMethod]
        public void DeleteTask(int id)
        {
            if (Application[TASKS] == null)
                { return; }

            int index = -1;
            String ID = id.ToString();
            List<Task> tasks = (List<Task>)Application[TASKS];
            for (int i = 0; i < tasks.Count; i++)
            {
                if (String.Equals(tasks[i].id, ID))
                {
                    index = i;
                    break;
                }
            }

            if (index == -1)
                { return; }

            if (String.Equals(tasks[index].state, "INITIALIZE")
                    || String.Equals(tasks[index].state, "FINISHED")
                        || String.Equals(tasks[index].state, "CRASHED"))
            {
                try
                {
                    String outputDir = (String)Application[RESULTS_FOLDER] + "/" + ID;
                    if (System.IO.Directory.Exists(outputDir))
                    { System.IO.Directory.Delete(outputDir, true); }

                    String inputDir = (String)Application[TASKS_FOLDER] + "/" + ID;
                    if (System.IO.Directory.Exists(inputDir))
                        { System.IO.Directory.Delete(inputDir, true); }
                }
                catch (Exception)
                {
                    System.Threading.Thread.Sleep(5000);
                    DeleteTask(id);
                }
            }
        }
    }
}