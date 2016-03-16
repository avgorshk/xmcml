using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace WEB_xmcml
{
    public partial class SURFACE : System.Web.UI.Page
    {
        protected void InitDropDownListSurfaces(int numLayers)
        {
            if (DropDownListSurfaces.Items.Count == 0)
            {
                for (int i = 1; i < numLayers + 1; i++)
                {
                    DropDownListSurfaces.Items.Add(i.ToString());
                }

                DropDownListSurfaces.SelectedIndex = 1;
            }
        }

        protected ServiceReferenceSurfaceCreator.ArrayOfString GetInfoMyTextBox(String textBoxName)
        {
            int numLayers = DropDownListSurfaces.SelectedIndex + 1;
            ServiceReferenceSurfaceCreator.ArrayOfString info
                = new ServiceReferenceSurfaceCreator.ArrayOfString();

            for (int i = 0; i < numLayers; i++)
            {
                int indexRows = -1, indexControls = -1;

                for (int j = TableSURFACE.Rows.Count - 1; j >= 0; j--)
                {
                    for (int k = 0; k < TableSURFACE.Rows[j].Cells[0].Controls.Count; k++)
                    {
                        if (String.Equals(TableSURFACE.Rows[j].Cells[0].Controls[k].ID, textBoxName + (i + 1).ToString()))
                        {
                            indexRows = j;
                            indexControls = k;
                            break;
                        }
                    }
                }

                if (indexRows == -1)
                {
                    info.Add("0.0");
                }
                else
                {
                    info.Add(((TextBox)TableSURFACE.Rows[indexRows].Cells[0].Controls[indexControls]).Text);
                }
            }
            return info;
        }

        protected void ClickButtonGetSURFACE(object sender, EventArgs e)
        {
            ServiceReferenceSurfaceCreator.ArrayOfString centerX = GetInfoMyTextBox("TextBoxCenterX");
            ServiceReferenceSurfaceCreator.ArrayOfString centerY = GetInfoMyTextBox("TextBoxCenterY");
            ServiceReferenceSurfaceCreator.ArrayOfString centerZ = GetInfoMyTextBox("TextBoxCenterZ");
            ServiceReferenceSurfaceCreator.ArrayOfString lengthX = GetInfoMyTextBox("TextBoxLengthX");
            ServiceReferenceSurfaceCreator.ArrayOfString lengthY = GetInfoMyTextBox("TextBoxLengthY");

            int numSurfaces = DropDownListSurfaces.SelectedIndex + 1;

            //Просим у сервера сделать SURFACE-файл и получаем его его адрес

            ServiceReferenceSurfaceCreator.Service1SoapClient serviceSURFACECreator = new ServiceReferenceSurfaceCreator.Service1SoapClient();
            String pathSURFACEFile = serviceSURFACECreator.GetPathSURFACEFile(numSurfaces,
                centerX, centerY, centerZ, lengthX, lengthY);

            if (pathSURFACEFile == null)
            {
                return;
            }

            //Передаем файл пользователю
            System.IO.FileInfo fileInfo = new System.IO.FileInfo(pathSURFACEFile);

            Context.Response.AddHeader("Content-Length", fileInfo.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "SURFACE-File/surface";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=mySURFACE.surface");
            Context.Response.WriteFile(pathSURFACEFile);
            Context.Response.End();
        }

        protected void AddTableEditorSurfaces(int numSurfaces)
        {
            TableRow row;
            TableCell cell;

            for (int i = 1; i < numSurfaces + 1; i++)
            {
                //Номер слоя
                row = new TableRow();

                Label labelNumerSurfaces = new Label();
                labelNumerSurfaces.ID = "LabelNumerSurfaces" + i.ToString();
                labelNumerSurfaces.Text = "Плоскость " + i.ToString() + ":";

                cell = new TableCell();
                cell.Controls.Add(labelNumerSurfaces);

                row.Cells.Add(cell);
                TableSURFACE.Rows.Add(row);

                //Описание поверхности
                //Первая строка
                row = new TableRow();

                Label labelCenter = new Label();
                labelCenter.ID = "LabelCenter" + i.ToString();
                labelCenter.Text = "Центр:";

                cell = new TableCell();
                cell.HorizontalAlign = HorizontalAlign.Center;
                cell.Controls.Add(labelCenter);

                row.Cells.Add(cell);
                TableSURFACE.Rows.Add(row);

                //Вторая строка
                row = new TableRow();

                Label labelCenterX = new Label();
                labelCenterX.ID = "LabelCenterX" + i.ToString();
                labelCenterX.Text = "X: ";

                TextBox textBoxCenterX = new TextBox();
                textBoxCenterX.ID = "TextBoxCenterX" + i.ToString();
                textBoxCenterX.Text = "0.0";
                textBoxCenterX.MaxLength = 25;

                Label labelCenterY = new Label();
                labelCenterY.ID = "LabelCenterY" + i.ToString();
                labelCenterY.Text = " Y: ";

                TextBox textBoxCenterY = new TextBox();
                textBoxCenterY.ID = "TextBoxCenterY" + i.ToString();
                textBoxCenterY.Text = "0.0";
                textBoxCenterY.MaxLength = 25;

                Label labelCenterZ = new Label();
                labelCenterZ.ID = "LabelCenterZ" + i.ToString();
                labelCenterZ.Text = " Z: ";

                TextBox textBoxCenterZ = new TextBox();
                textBoxCenterZ.ID = "TextBoxCenterZ" + i.ToString();
                textBoxCenterZ.Text = "0.0";
                textBoxCenterZ.MaxLength = 25;

                cell = new TableCell();
                cell.HorizontalAlign = HorizontalAlign.Center;

                cell.Controls.Add(labelCenterX);
                cell.Controls.Add(textBoxCenterX);
                cell.Controls.Add(labelCenterY);
                cell.Controls.Add(textBoxCenterY);
                cell.Controls.Add(labelCenterZ);
                cell.Controls.Add(textBoxCenterZ);

                row.Cells.Add(cell);
                TableSURFACE.Rows.Add(row);

                //Третья строка
                row = new TableRow();

                Label labelLength = new Label();
                labelLength.ID = "LabelLength" + i.ToString();
                labelLength.Text = "Длина:";

                cell = new TableCell();
                cell.HorizontalAlign = HorizontalAlign.Center;
                cell.Controls.Add(labelLength);

                row.Cells.Add(cell);
                TableSURFACE.Rows.Add(row);

                //Четвертая строка
                row = new TableRow();

                Label labelLengthX = new Label();
                labelLengthX.ID = "LabelLengthX" + i.ToString();
                labelLengthX.Text = "X: ";

                TextBox textBoxLengthX = new TextBox();
                textBoxLengthX.ID = "TextBoxLengthX" + i.ToString();
                textBoxLengthX.Text = "0.0";
                textBoxLengthX.MaxLength = 25;

                Label labelLengthY = new Label();
                labelLengthY.ID = "LabelLengthY" + i.ToString();
                labelLengthY.Text = " Y: ";

                TextBox textBoxLengthY = new TextBox();
                textBoxLengthY.ID = "TextBoxLengthY" + i.ToString();
                textBoxLengthY.Text = "0.0";
                textBoxLengthY.MaxLength = 25;

                cell = new TableCell();
                cell.HorizontalAlign = HorizontalAlign.Center;

                cell.Controls.Add(labelLengthX);
                cell.Controls.Add(textBoxLengthX);
                cell.Controls.Add(labelLengthY);
                cell.Controls.Add(textBoxLengthY);

                row.Cells.Add(cell);
                TableSURFACE.Rows.Add(row);

                //Пустая строка
                row = new TableRow();
                cell = new TableCell();
                cell.Text = "&nbsp";

                row.Cells.Add(cell);
                TableSURFACE.Rows.Add(row);
            }

            row = new TableRow();

            Button buttonGetSURFACE = new Button();
            buttonGetSURFACE.ID = "ButtonGetSurfaces";
            buttonGetSURFACE.Text = "Получить SURFACE";
            buttonGetSURFACE.Click += new EventHandler(ClickButtonGetSURFACE);

            cell = new TableCell();
            cell.HorizontalAlign = HorizontalAlign.Center;
            cell.Controls.Add(buttonGetSURFACE);

            row.Cells.Add(cell);
            TableSURFACE.Rows.Add(row);
        }

        protected void Page_Load(object sender, EventArgs e)
        {
            const int NUM_LAYERS = 4;
            InitDropDownListSurfaces(NUM_LAYERS);
            AddTableEditorSurfaces(DropDownListSurfaces.SelectedIndex + 1);
        }
    }
}