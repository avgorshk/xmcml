using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace WEB_xmcml
{
    public partial class tasks : System.Web.UI.Page
    {
        ServiceReferenceControlTasks.Service1SoapClient service;

        protected void CreateTable()
        {
            const int numTextColumns = 4;
            const int numControlColumns = 4;
            System.Drawing.Color myHeaderCellColor = System.Drawing.Color.FromName("#9999FF");

            int numTasks = service.GetNumTasks();
            ServiceReferenceControlTasks.ArrayOfString tableStr = service.GetTableStr(DropDownSort.SelectedIndex);

            TableRow row;
            TableCell[] cells = new TableCell[numTextColumns + numControlColumns];

            row = new TableRow();
            cells[0] = new TableCell();
            cells[0].ColumnSpan = numTextColumns + numControlColumns;
            cells[0].Text = "Всего задач на сервере: " + numTasks.ToString();

            row.Cells.Add(cells[0]);
            TableTasks.Rows.Add(row);

            row = new TableRow();

            for (int i = 0; i < numTextColumns; i++)
            {
                cells[i] = new TableCell();
                cells[i].Font.Bold = true;
                cells[i].BorderStyle = BorderStyle.Solid;
                cells[i].BorderColor = System.Drawing.Color.FromArgb(0, 0, 0);
                cells[i].BorderWidth = Unit.Point(1);
                cells[i].BackColor = myHeaderCellColor;
            }

            for (int i = numTextColumns; i < numTextColumns + numControlColumns; i++)
            {
                cells[i] = new TableCell();
            }

            cells[0].Text = "ID";
            cells[0].Width = Unit.Percentage(5);
            cells[1].Text = "Имя задачи";
            cells[2].Width = Unit.Percentage(50);
            cells[2].Text = "Дата и время";
            cells[2].Width = Unit.Percentage(25);
            cells[3].Text = "Статус";
            cells[3].Width = Unit.Percentage(20);

            for (int i = numTextColumns; i < numTextColumns + numControlColumns; i++)
            {
                cells[i].Width = Unit.Pixel(1);
            }

            for (int i = 0; i < numTextColumns + numControlColumns; i++)
            {
                row.Cells.Add(cells[i]);
            }

            row.Height = Unit.Point(30);
            TableTasks.Rows.Add(row);
            
            for (int i = 0; i < numTasks; i++)
            {
                row = new TableRow();

                for (int j = 0; j < numTextColumns; j++)
                {
                    cells[j] = new TableCell();
                    cells[j].BorderStyle = BorderStyle.Solid;
                    cells[j].BorderColor = System.Drawing.Color.FromArgb(0, 0, 0);
                    cells[j].BorderWidth = Unit.Point(1);
                }

                cells[0].Text = tableStr[i * numTextColumns];
                cells[1].Text = tableStr[i * numTextColumns + 1];
                cells[2].Text = tableStr[i * numTextColumns + 2];
                cells[3].Text = tableStr[i * numTextColumns + 3];

                for (int j = numTextColumns; j < numTextColumns + numControlColumns; j++)
                {
                    cells[j] = new TableCell();
                    cells[j].BorderStyle = BorderStyle.Solid;
                    cells[j].BorderColor = System.Drawing.Color.FromArgb(0, 0, 0);
                    cells[j].BorderWidth = Unit.Point(1);
                    cells[j].BackColor = myHeaderCellColor;
                }

                Button buttonDownloadOUT = new Button();

                buttonDownloadOUT.ID = "OUT" + cells[0].Text;
                buttonDownloadOUT.Text = "stdout";
                buttonDownloadOUT.Click += new EventHandler(ClickDownloadButtonOUT);
                buttonDownloadOUT.Enabled = false;

                Button buttonDownloadMCML = new Button();

                buttonDownloadMCML.ID = "MCML" + cells[0].Text;
                buttonDownloadMCML.Text = "mcml";
                buttonDownloadMCML.Click += new EventHandler(ClickDownloadButtonMCML);
                buttonDownloadMCML.Enabled = false;

                Button buttonView = new Button();

                buttonView.ID = "VIEW" + cells[0].Text;
                buttonView.Text = "O";
                buttonView.Click += new EventHandler(ClickViewButton);
                buttonView.Enabled = false;

                Button buttonDelete = new Button();

                buttonDelete.ID = "DELETE" + cells[0].Text;
                buttonDelete.Text = "X";
                buttonDelete.Click += new EventHandler(ClickDeleteButton);
                buttonDelete.Enabled = false;

                if(String.Equals(cells[3].Text, "Завершено") || String.Equals(cells[3].Text, "Ошибка"))
                {
                    buttonDownloadOUT.Enabled = true;

                    if (String.Equals(cells[3].Text, "Завершено"))
                    {
                        buttonDownloadMCML.Enabled = true;
                    }
                }
                if (String.Equals(cells[3].Text, "Завершено"))
                {
                    buttonView.Enabled = true;
                }
                if (String.Equals(cells[3].Text, "Инициализация")
                    || String.Equals(cells[3].Text, "Завершено")
                        || String.Equals(cells[3].Text, "Ошибка"))
                {
                    buttonDelete.Enabled = true;
                }

                cells[4].Controls.Add(buttonDownloadOUT);
                cells[5].Controls.Add(buttonDownloadMCML);
                cells[6].Controls.Add(buttonView);
                cells[7].Controls.Add(buttonDelete);

                for (int j = 0; j < numTextColumns + numControlColumns; j++)
                {
                    row.Cells.Add(cells[j]);
                }

                TableTasks.Rows.Add(row);
            }
        }

        protected void InitDropDownSort()
        {
            if (DropDownSort.Items.Count == 0)
            {
                DropDownSort.Items.Add("ID");
                DropDownSort.Items.Add("имени");
                DropDownSort.Items.Add("времени");
                DropDownSort.Items.Add("статусу");
            }
        }

        protected void Page_Load(object sender, EventArgs e)
        {
            service = new ServiceReferenceControlTasks.Service1SoapClient();
            service.Init();

            InitDropDownSort();
            CreateTable();
        }

        protected void ClickDownloadButtonOUT(object sender, EventArgs e)
        {
            int id = Convert.ToInt32(((Button)sender).ID.Replace("OUT", ""));

            String filePath = service.GetPathDownloadOUT(id);

            if (filePath == null)
            {
                return;
            }

            System.IO.FileInfo file = new System.IO.FileInfo(filePath);

            Context.Response.AddHeader("Content-Length", file.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "text/txt";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + file.Name);
            Context.Response.WriteFile(file.FullName);
            Context.Response.End();
        }

        protected void ClickDownloadButtonMCML(object sender, EventArgs e)
        {
            int id = Convert.ToInt32(((Button)sender).ID.Replace("MCML", ""));

            String filePath = service.GetPathDownloadMCML(id);

            if (filePath == null)
            {
                return;
            }

            System.IO.FileInfo file = new System.IO.FileInfo(filePath);
            
            Context.Response.AddHeader("Content-Length", file.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "MCMLFile/mcml.output";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + file.Name);       
            Context.Response.WriteFile(file.FullName);
            Context.Response.End();
        }

        protected void ClickViewButton(object sender, EventArgs e)
        {
            int id = Convert.ToInt32(((Button)sender).ID.Replace("VIEW", ""));
            Response.Redirect("MCMLViewer.aspx?id=" + id.ToString());
        }

        protected void ClickDeleteButton(object sender, EventArgs e)
        {
            int id = Convert.ToInt32(((Button)sender).ID.Replace("DELETE", ""));
            service.DeleteTask(id);
            Response.Redirect(Request.Path);
        }
    }
}