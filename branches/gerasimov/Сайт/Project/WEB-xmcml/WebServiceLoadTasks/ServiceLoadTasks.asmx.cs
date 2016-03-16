using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;

namespace WebServiceLoadTasks
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
        private const String CURRENT_ID = "CURRENT_ID";
        private const String TASKS_FOLDER = "TASKS_FOLDER";
        private const String FILE_CONFIG = "FILE_CONFIG";

        private const String FILE_PATHS = "D://PATHS.cfg";

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

        private void ParseConfig(String config)
        {
            Char[] separator = { '=', '\r', '\n' };
            String[] info = config.Split(separator, StringSplitOptions.RemoveEmptyEntries);

            for (int i = 0; i < info.Length; i += 2)
            {
                switch (info[i])
                {
                    case "CUR_ID":
                        {
                            Application[CURRENT_ID] = Convert.ToInt32(info[i + 1]);
                            break;
                        }
                }
            }

            if (Application[CURRENT_ID] == null)
            {
                Application[CURRENT_ID] = -1;
            }
        }

        [WebMethod]
        public void Init()
        {
            String pathFileConfig;

            Application[TASKS_FOLDER] = GetValueFromPaths("TASKS_FOLDER");
            Application[FILE_CONFIG] = pathFileConfig = GetValueFromPaths("FILE_CONFIG");

            if (!System.IO.File.Exists(pathFileConfig))
            {
                Application[CURRENT_ID] = -1;
            }
            else
            {
                System.IO.StreamReader reader = new System.IO.StreamReader(pathFileConfig);
                String config = reader.ReadToEnd();
                reader.Close();
                ParseConfig(config);
            }
        }

        [WebMethod]
        public int GetCurrentID()
        {
            return (int)Application[CURRENT_ID];
        }

        [WebMethod]
        public void ResetID()
        {
            Application[CURRENT_ID] = -1;
        }

        private void RewriteConfig()
        {
            String config = "CUR_ID=" + Application[CURRENT_ID].ToString() + "\r\n";

            if (Application[FILE_CONFIG] == null)
            {
                return;
            }

            String pathFileConfig = (String)Application[FILE_CONFIG];

            System.IO.StreamWriter writer = new System.IO.StreamWriter(pathFileConfig);
            writer.Write(config);
            writer.Close();
        }

        private void CreateInfoFile(String taskName)
        {
            String dirName = (String)Application[TASKS_FOLDER] + "/" + ((int)(Application[CURRENT_ID])).ToString();

            if (System.IO.Directory.Exists(dirName))
            {
                System.IO.Directory.Delete(dirName, true);
            }
            System.IO.Directory.CreateDirectory(dirName);
            
            System.IO.StreamWriter writer = new System.IO.StreamWriter(dirName + "/INFO.TXT");
            writer.WriteLine("ID=" + ((int)(Application[CURRENT_ID])).ToString());
            writer.WriteLine("NAME=" + taskName);
            writer.WriteLine("DATE_AND_TIME=" + System.DateTime.Now.ToString());
            writer.WriteLine("STATE=INITIALIZE");
            writer.Close();
        }

        [WebMethod]
        public int GetIDNewTask(String taskName)
        {
            Application[CURRENT_ID] = (int)Application[CURRENT_ID] + 1;

            CreateInfoFile(taskName);
            RewriteConfig();

            return (int)Application[CURRENT_ID];
        }

        [WebMethod]
        public String GetPathXML(int id)
        {
            return (String)Application[TASKS_FOLDER] + "/" + id.ToString() + "/" + id.ToString() + ".XML";
        }

        [WebMethod]
        public String GetPathSURFACE(int id)
        {
            return (String)Application[TASKS_FOLDER] + "/" + id.ToString() + "/" + id.ToString() + ".SURFACE";
        }
    }
}