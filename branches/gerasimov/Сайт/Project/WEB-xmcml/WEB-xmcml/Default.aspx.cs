using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace WEB_xmcml
{
    public partial class _Default : System.Web.UI.Page
    {
        ServiceReferenceLoadTasks.Service1SoapClient service;

        protected void Page_Load(object sender, EventArgs e)
        {
            service = new ServiceReferenceLoadTasks.Service1SoapClient();
            service.Init();
        }

        protected void ButtonSend_Click(object sender, EventArgs e)
        {
            LabelError.Text = "";

            if ((!FileUploadSURFACE.HasFile) || (!FileUploadXML.HasFile))
            {
                LabelError.Text = "Не все файлы выбраны для загрузки!";
            }
            else
            {
                String[] tmpStr;
                String extension;

                tmpStr = FileUploadXML.FileName.Split('.');
                extension = tmpStr[tmpStr.Length - 1];

                if(!String.Equals(extension, "xml", StringComparison.InvariantCultureIgnoreCase))
                {
                    LabelError.Text = "Проверьте XML файл!";
                    return;
                }

                tmpStr = FileUploadSURFACE.FileName.Split('.');
                extension = tmpStr[tmpStr.Length - 1];

                if(!String.Equals(extension, "surface", StringComparison.InvariantCultureIgnoreCase))
                {
                    LabelError.Text = "Проверьте SURFACE файл!";
                    return;
                }

                if (TextBoxNameTasks.Text.Length < 4)
                {
                    LabelError.Text = "Имя задачи должно быть более 3-ёх символов!";
                    return;
                }

                if (TextBoxNameTasks.Text.IndexOf(' ') >= 0)
                {
                    LabelError.Text = "Имя задачи должно быть без пробелов!";
                    return;
                }

                int id = service.GetIDNewTask(TextBoxNameTasks.Text);
                FileUploadXML.SaveAs(service.GetPathXML(id));
                FileUploadSURFACE.SaveAs(service.GetPathSURFACE(id));
            }
        }
    }
}