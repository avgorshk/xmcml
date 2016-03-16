using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace WEB_xmcml
{
    public partial class MCMLViewer : System.Web.UI.Page
    {
        private const String TASK_ID = "TASK_ID";
        private const String UNICAL_SESSION_ID = "UNICAL_SESSION_ID";

        ServiceReferenceMCMLViewer.Service1SoapClient serviceMCMLViewer;

        private const int WIDTH_CLIENT_REGION = 300;
        private const int HEIGHT_CLIENT_REGION = 300;
        private const String PAGE_NAME = "MCMLViewer.aspx";

        public String GetUnicalSessionID()
        {
            System.DateTime now = System.DateTime.Now;

            Random rand = new Random(now.Millisecond + now.Second + now.Minute + now.Hour);

            String UnicalSessionID = now.Day.ToString() + "_" + now.Month.ToString()
                + "_" + now.Year.ToString() + "key=" + rand.Next().ToString();

            return UnicalSessionID;
        }
   
        private void InitDropDownListMode()
        {
            if (DropDownListMode.Items.Count == 0)
            {
                DropDownListMode.Items.Add("Карта поглощения");
                DropDownListMode.Items.Add("Карта рассеивания");
                DropDownListMode.Items.Add("Глубинная карта");
                DropDownListMode.Items.Add("Детекторы");
                DropDownListMode.Items.Add("Сетка детекторов");
                DropDownListMode.SelectedIndex = 0;
            }
        }

        private void InitDropDownListDetectorID()
        {
            if (DropDownListMode.SelectedIndex != 3)
            {
                LabelDetectorID.Visible = false;
                DropDownListDetectorID.Visible = false;
            }
            else
            {
                LabelDetectorID.Visible = true;
                DropDownListDetectorID.Visible = true;
            }

            if (DropDownListDetectorID.Items.Count == 0)
            {
                int numberOfDetectors = serviceMCMLViewer.GetNumberOfDetectors();
                if (numberOfDetectors == 0)
                    { return; }
                else
                {
                    for (int i = 0; i < numberOfDetectors; i++)
                        { DropDownListDetectorID.Items.Add(i.ToString()); }
                    DropDownListDetectorID.SelectedIndex = 0;
                }
            }
        }

        protected void OnClickButtonOriginal(object sender, EventArgs e)
        {
            String type = ((Button)sender).ID.Replace("BUTTON_ORIGINAL", "");
            if ((type != "XY") && (DropDownListMode.SelectedIndex == 4))
                { return; }

            String pathImageOriginal = serviceMCMLViewer.GetPathImageOriginal((String)Session[UNICAL_SESSION_ID], type);
            if (pathImageOriginal == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathImageOriginal);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "IMAGE/img";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + fileInfo.Name);
            Context.Response.WriteFile(pathImageOriginal);
            Context.Response.End();
        }

        protected void OnClickButtonMatrix(object sender, EventArgs e)
        {
            String type = ((Button)sender).ID.Replace("BUTTON_MATRIX", "");
            if ((type != "XY") && (DropDownListMode.SelectedIndex == 4))
                { return; }

            String pathMatrix = serviceMCMLViewer.GetPathMatrix((String)Session[UNICAL_SESSION_ID], type);
            if (pathMatrix == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathMatrix);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "TEXT/txt";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + fileInfo.Name);
            Context.Response.WriteFile(pathMatrix);
            Context.Response.End();
        }

        protected void OnClickButtonSaveAll(object sender, EventArgs e)
        {
            String pathAllAsText = serviceMCMLViewer.GetPathAllAsText((String)Session[UNICAL_SESSION_ID]);

            if (pathAllAsText == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathAllAsText);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "TEXT/txt";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + fileInfo.Name);
            Context.Response.WriteFile(pathAllAsText);
            Context.Response.End();
        }

        protected void OnClickButtonSaveTimeScales(object sender, EventArgs e)
        {
            String pathTimeScales= serviceMCMLViewer.GetPathTimeScales((String)Session[UNICAL_SESSION_ID]);

            if (pathTimeScales == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathTimeScales);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "TEXT/txt";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + fileInfo.Name);
            Context.Response.WriteFile(pathTimeScales);
            Context.Response.End();
        }

        protected void OnClickButtonSaveWeightInDetectors(object sender, EventArgs e)
        {
            int mode;

            if (RadioButtonWD1.Checked)
            {
                mode = 0;
            }
            else if (RadioButtonWD2.Checked)
            {
                mode = 1;
            }
            else if (RadioButtonWD3.Checked)
            {
                mode = 2;
            }
            else
            {
                return;
            }

            String pathWeightInDetectors =
                serviceMCMLViewer.GetPathWeights((String)Session[UNICAL_SESSION_ID], mode);

            if (pathWeightInDetectors == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathWeightInDetectors);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "TEXT/txt";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + fileInfo.Name);
            Context.Response.WriteFile(pathWeightInDetectors);
            Context.Response.End();
        }
     
        protected void OnClickButtonSaveDetectorsRanges(object sender, EventArgs e)
        {
            int mode;

            if (RadioButtonDRI1.Checked)
            {
                mode = 0;
            }
            else if (RadioButtonDRI2.Checked)
            {
                mode = 1;
            }
            else if (RadioButtonDRI3.Checked)
            {
                mode = 2;
            }
            else if (RadioButtonDRI4.Checked)
            {
                mode = 3;
            }
            else
            {
                return;
            }

            String pathDetectorsRangesInfo =
                serviceMCMLViewer.GetPathRanges((String)Session[UNICAL_SESSION_ID], mode);

            if (pathDetectorsRangesInfo == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathDetectorsRangesInfo);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "TEXT/txt";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=" + fileInfo.Name);
            Context.Response.WriteFile(pathDetectorsRangesInfo);
            Context.Response.End();
        }

        protected void InitSelectors()
        {
            InitSelector("XY");
            InitSelector("XZ");
            InitSelector("YZ");
        }

        protected void InitSelector(String type)
        {
            DropDownList dropDownListBox;
            switch (type)
            {
                case "XY":
                    {
                        dropDownListBox = DropDownListXY;
                        break;
                    }
                case "XZ":
                    {
                        dropDownListBox = DropDownListXZ;
                        break;
                    }
                case "YZ":
                    {
                        dropDownListBox = DropDownListYZ;
                        break;
                    }
                default: { return; }
            }

            if (dropDownListBox.Items.Count != 0)
                { return; }

            ServiceReferenceMCMLViewer.ArrayOfDouble infoOfArea = serviceMCMLViewer.GetInfoOfArea(type);
            if (infoOfArea == null) { return; }

            int partition = (int)infoOfArea[2];
            double corner = infoOfArea[0];
            double length = infoOfArea[1];

            if (partition > 0)
            {
                if (partition == 1)
                {
                    dropDownListBox.Items.Add((corner + length / 2.0).ToString());
                }
                else
                {
                    for (int i = 0; i < partition; i++)
                    {
                        dropDownListBox.Items.Add((corner + i * length / partition).ToString());
                    }
                    dropDownListBox.Items.Add((corner + length).ToString());
                }
                dropDownListBox.SelectedIndex = (int)(partition / 2.0);
            }
        }

        protected void AddImages()
        {
            double var;
            String pathImage;

            var = Convert.ToDouble(DropDownListXY.Items[DropDownListXY.SelectedIndex].Text);
            serviceMCMLViewer.InitFiles((String)Session[UNICAL_SESSION_ID], WIDTH_CLIENT_REGION, HEIGHT_CLIENT_REGION, "XY", var);
            pathImage = serviceMCMLViewer.GetNameImageLogo((String)Session[UNICAL_SESSION_ID], "XY");
            
            if (pathImage == null)
            {
                LabelXY_Img.Text = "<img src=\"../img/no_image.jpg\" width=\"" + WIDTH_CLIENT_REGION +
                    "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
            }
            else
            {
                pathImage = "../img/VIEWER/" + pathImage;
                LabelXY_Img.Text = "<img src=\"" + pathImage + "\" width=\"" + WIDTH_CLIENT_REGION
                    + "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
            }

            if (DropDownListMode.SelectedIndex == 4)
            {
                LabelXZ_Img.Text = "<img src=\"../img/no_image.jpg\" width=\"" + WIDTH_CLIENT_REGION +
                    "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
                LabelYZ_Img.Text = "<img src=\"../img/no_image.jpg\" width=\"" + WIDTH_CLIENT_REGION +
                    "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
            }
            else
            {
                var = Convert.ToDouble(DropDownListXZ.Items[DropDownListXZ.SelectedIndex].Text);
                serviceMCMLViewer.InitFiles((String)Session[UNICAL_SESSION_ID], WIDTH_CLIENT_REGION, HEIGHT_CLIENT_REGION, "XZ", var);
                pathImage = serviceMCMLViewer.GetNameImageLogo((String)Session[UNICAL_SESSION_ID], "XZ");
                if (pathImage == null)
                {
                    LabelXZ_Img.Text = "<img src=\"../img/no_image.jpg\" width=\"" + WIDTH_CLIENT_REGION +
                        "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
                }
                else
                {
                    pathImage = "../img/VIEWER/" + pathImage;
                    LabelXZ_Img.Text = "<img src=\"" + pathImage + "\" width=\"" + WIDTH_CLIENT_REGION
                        + "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
                }

                var = Convert.ToDouble(DropDownListYZ.Items[DropDownListYZ.SelectedIndex].Text);
                serviceMCMLViewer.InitFiles((String)Session[UNICAL_SESSION_ID], WIDTH_CLIENT_REGION, HEIGHT_CLIENT_REGION, "YZ", var);
                pathImage = serviceMCMLViewer.GetNameImageLogo((String)Session[UNICAL_SESSION_ID], "YZ");
                if (pathImage == null)
                {
                    LabelYZ_Img.Text = "<img src=\"../img/no_image.jpg\" width=\"" + WIDTH_CLIENT_REGION +
                        "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
                }
                else
                {
                    pathImage = "../img/VIEWER/" + pathImage;
                    LabelYZ_Img.Text = "<img src=\"" + pathImage + "\" width=\"" + WIDTH_CLIENT_REGION
                        + "px\" height=\"" + HEIGHT_CLIENT_REGION + "px\"/ ></img>\n";
                }
            }
        }

        protected void ReadQueryID()
        {
            String id = Request.QueryString["id"];
            if (Session[TASK_ID] != null)
            {
                if ((String)Session[TASK_ID] != id)
                {
                    Session[UNICAL_SESSION_ID] = null;
                    DropDownListDetectorID.Items.Clear();
                }
            }
            Session[TASK_ID] = id;
        }

        protected void Page_Load(object sender, EventArgs e)
        {
            serviceMCMLViewer = new ServiceReferenceMCMLViewer.Service1SoapClient();
            InitDropDownListMode();
            InitDropDownListDetectorID();

            int mode = DropDownListMode.SelectedIndex;
            ReadQueryID();

            int detectorID = DropDownListDetectorID.SelectedIndex;
            bool isOk = serviceMCMLViewer.OpenMCML((String)Session[TASK_ID], mode, detectorID);
            if (!isOk)
            {
                LabelInfo.Text = "Файл для просмотра не выбран!";
                TableMCMLViewer.Visible = false;
            }
            else
            {
                LabelInfo.Text = "Информация:";
                TableMCMLViewer.Visible = true;
                TextBoxInfo.Text = serviceMCMLViewer.GetInfoMCML();

                if (Session[UNICAL_SESSION_ID] == null)
                {
                    Session[UNICAL_SESSION_ID] = GetUnicalSessionID();
                }
                InitSelectors();
                AddImages();
            }
        }
    }
}