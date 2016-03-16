using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.UI;
using System.Web.UI.WebControls;

namespace WEB_xmcml
{
    public partial class XML : System.Web.UI.Page
    {
        protected void InitDropDownListLayers(int numLayers)
        {
            if (DropDownListLayers.Items.Count == 0)
            {
                for (int i = 1; i < numLayers + 1; i++)
                {
                    DropDownListLayers.Items.Add(i.ToString());
                }

                DropDownListLayers.SelectedIndex = 1;
            }
        }

        protected ServiceReferenceXMLCreator.ArrayOfString GetInfoMyTextBox(String textBoxName)
        {
            int numLayers = DropDownListLayers.SelectedIndex + 1;
            ServiceReferenceXMLCreator.ArrayOfString info
                = new ServiceReferenceXMLCreator.ArrayOfString();

            for (int i = 0; i < numLayers; i++)
            {
                int indexRows = -1, indexControls = -1;

                for (int j = TableXML.Rows.Count - 1; j >= 0; j--)
                {
                    for (int k = 0; k < TableXML.Rows[j].Cells[0].Controls.Count; k++)
                    {
                        if (String.Equals(TableXML.Rows[j].Cells[0].Controls[k].ID, textBoxName + (i + 1).ToString()))
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
                    info.Add(((TextBox)TableXML.Rows[indexRows].Cells[0].Controls[indexControls]).Text);
                }
            }
            return info;
        }

        protected void ClickButtonGetXML(object sender, EventArgs e)
        {
            String numberOfPhotons = TextBoxNumberPhotons.Text;
            String minWeight = TextBoxMinWeight.Text;

            ServiceReferenceXMLCreator.ArrayOfString corner
                = new ServiceReferenceXMLCreator.ArrayOfString();
            corner.Add(TextBoxCornerX.Text);
            corner.Add(TextBoxCornerY.Text);
            corner.Add(TextBoxCornerZ.Text);

            ServiceReferenceXMLCreator.ArrayOfString length
                = new ServiceReferenceXMLCreator.ArrayOfString();
            length.Add(TextBoxLengthX.Text);
            length.Add(TextBoxLengthY.Text);
            length.Add(TextBoxLengthZ.Text);

            ServiceReferenceXMLCreator.ArrayOfString partition
                = new ServiceReferenceXMLCreator.ArrayOfString();
            partition.Add(TextBoxPartitionX.Text);
            partition.Add(TextBoxPartitionY.Text);
            partition.Add(TextBoxPartitionZ.Text);

            ServiceReferenceXMLCreator.ArrayOfString refractiveIndex = GetInfoMyTextBox("TextBoxRefractiveIndex");
            ServiceReferenceXMLCreator.ArrayOfString absorptionCoefficient = GetInfoMyTextBox("TextBoxAbsorptionCoefficient");
            ServiceReferenceXMLCreator.ArrayOfString scatteringCoefficient = GetInfoMyTextBox("TextBoxScatteringCoefficient");
            ServiceReferenceXMLCreator.ArrayOfString anisotropy = GetInfoMyTextBox("TextBoxAnisotropy");

            int numLayers = DropDownListLayers.SelectedIndex + 1;

            //Просим у сервера сделать XML-текст
            ServiceReferenceXMLCreator.Service1SoapClient serviceXMLCreator = new ServiceReferenceXMLCreator.Service1SoapClient();
            String XML = serviceXMLCreator.GetXML(numberOfPhotons, minWeight, corner, length, partition,
                numLayers, refractiveIndex, absorptionCoefficient, scatteringCoefficient, anisotropy);

            //Передаем файл пользователю
            Context.Response.AddHeader("Content-Length", XML.Length.ToString());
            Context.Response.AddHeader("Connection", "Keep-Alive");
            Context.Response.ContentType = "XML-File/xml";
            Context.Response.AddHeader("Content-Disposition", "attachment;filename=myXML.xml");
            Context.Response.Write(XML);
            Context.Response.End();
        }

        protected void AddTableEditorLayers(int numLayers)
        {
            TableRow row;
            TableCell cell;

            for (int i = 1; i < numLayers + 1; i++)
            {
                //Номер слоя
                row = new TableRow();

                Label labelNumerLayers = new Label();
                labelNumerLayers.ID = "LabelNumerLayers" + i.ToString();
                labelNumerLayers.Text = "Слой " + i.ToString() + ":";

                cell = new TableCell();
                cell.Controls.Add(labelNumerLayers);

                row.Cells.Add(cell);
                TableXML.Rows.Add(row);

                //Начало описания слоя
                //Первая строка
                row = new TableRow();

                Label labelRefractiveIndex = new Label();
                labelRefractiveIndex.ID = "LabelRefractiveIndex" + i.ToString();
                labelRefractiveIndex.Text = "Коэффициент преломления: ";

                TextBox textBoxRefractiveIndex = new TextBox();
                textBoxRefractiveIndex.ID = "TextBoxRefractiveIndex" + i.ToString();
                textBoxRefractiveIndex.Text = "0.0";
                textBoxRefractiveIndex.MaxLength = 25;

                Label labelAbsorptionCoefficient = new Label();
                labelAbsorptionCoefficient.ID = "LabelAbsorptionCoefficient" + i.ToString();
                labelAbsorptionCoefficient.Text = " Коэффициент поглощения: ";

                TextBox textBoxAbsorptionCoefficient = new TextBox();
                textBoxAbsorptionCoefficient.ID = "TextBoxAbsorptionCoefficient" + i.ToString();
                textBoxAbsorptionCoefficient.Text = "0.0";
                textBoxAbsorptionCoefficient.MaxLength = 25;

                cell = new TableCell();
                cell.HorizontalAlign = HorizontalAlign.Center;
                
                cell.Controls.Add(labelRefractiveIndex);
                cell.Controls.Add(textBoxRefractiveIndex);
                cell.Controls.Add(labelAbsorptionCoefficient);
                cell.Controls.Add(textBoxAbsorptionCoefficient);

                row.Cells.Add(cell);
                TableXML.Rows.Add(row);

                //Вторая строка
                row = new TableRow();

                Label labelScatteringCoefficient = new Label();
                labelScatteringCoefficient.ID = "LabelScatteringCoefficient" + i.ToString();
                labelScatteringCoefficient.Text = "Коэффициент рассеивания: ";

                TextBox textBoxScatteringCoefficient = new TextBox();
                textBoxScatteringCoefficient.ID = "TextBoxScatteringCoefficient" + i.ToString();
                textBoxScatteringCoefficient.Text = "0.0";
                textBoxScatteringCoefficient.MaxLength = 25;

                Label labelAnisotropy = new Label();
                labelAnisotropy.ID = "LabelAnisotropy" + i.ToString();
                labelAnisotropy.Text = " Коэффициент анизотропии: ";

                TextBox textBoxAnisotropy = new TextBox();
                textBoxAnisotropy.ID = "TextBoxAnisotropy" + i.ToString();
                textBoxAnisotropy.Text = "0.0";
                textBoxAnisotropy.MaxLength = 25;

                cell = new TableCell();
                cell.HorizontalAlign = HorizontalAlign.Center;

                cell.Controls.Add(labelScatteringCoefficient);
                cell.Controls.Add(textBoxScatteringCoefficient);
                cell.Controls.Add(labelAnisotropy);
                cell.Controls.Add(textBoxAnisotropy);

                row.Cells.Add(cell);
                TableXML.Rows.Add(row);

                //Пустая строка
                row = new TableRow();
                cell = new TableCell();
                cell.Text = "&nbsp";

                row.Cells.Add(cell);
                TableXML.Rows.Add(row);
            }

            row = new TableRow();

            Button buttonGetXML = new Button();
            buttonGetXML.ID = "ButtonGetXML";
            buttonGetXML.Text = "Получить XML";
            buttonGetXML.Click += new EventHandler(ClickButtonGetXML);

            cell = new TableCell();
            cell.HorizontalAlign = HorizontalAlign.Center;
            cell.Controls.Add(buttonGetXML);

            row.Cells.Add(cell);
            TableXML.Rows.Add(row);
        }

        protected void Page_Load(object sender, EventArgs e)
        {
            const int NUM_LAYERS = 5;
            InitDropDownListLayers(NUM_LAYERS);
            AddTableEditorLayers(DropDownListLayers.SelectedIndex + 1);
        }
    }
}