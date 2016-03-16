<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="MCMLViewer.aspx.cs" Inherits="WEB_xmcml.MCMLViewer" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title>MCML-Viewer</title>
    <link rel="Stylesheet" href="../css/WEB-MCML_STYLE.css" />
</head>
<body>
    <form id="form1" runat="server">

    <div align="center">
    <asp:Table ID="TableHeader" runat="server" Width="75%" BorderColor="#000000" BorderWidth="1px"
         BackImageUrl="../img/header.jpg">
        <asp:TableRow>
            <asp:TableCell ColumnSpan="4" HorizontalAlign="Right" VerticalAlign="Middle">
                <h1>WEB-MCML</h1>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow BackColor="#9999FF">
            <asp:TableCell Width="25%">
                <a href="Default.aspx" >Главная</a>
            </asp:TableCell>
            <asp:TableCell Width="25%">
                <a href="Tasks.aspx">Задачи</a>
            </asp:TableCell>
            <asp:TableCell Width="25%">
                <a href="XML.aspx">XML-генератор</a>
            </asp:TableCell>
            <asp:TableCell Width="25%">
                <a href="SURFACE.aspx">SURFACE-генератор</a>
            </asp:TableCell>
        </asp:TableRow>
    </asp:Table>
    </div>

    <div align="center">
    <asp:Table ID="TableMCMLInfo" runat="server" Width="75%" BorderColor="#000000" BorderWidth="1px">
        <asp:TableRow>
            <asp:TableCell>
                <asp:Label ID="LabelInfo" runat="server" Text=""></asp:Label>
            </asp:TableCell>
        </asp:TableRow>
    </asp:Table>
    <asp:Table ID="TableMCMLViewer" runat="server" Width="75%" BorderColor="#000000" BorderWidth="1px">
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:TextBox ID="TextBoxInfo" Width="97%" Height="200px" runat="server" TextMode="MultiLine" ReadOnly="True">
                </asp:TextBox>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Button ID="ButtonSaveWeightInDetectors" runat="server" Text="Сохранить веса на детекторах (*.txt)"
                    OnClick="OnClickButtonSaveWeightInDetectors" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelWD1" runat="server" Text="С нормировкой: "></asp:Label>
                <asp:RadioButton ID="RadioButtonWD1" GroupName="WD" runat="server" />
                <asp:Label ID="LabelWD2" runat="server" Text="| Без нормировки: "></asp:Label>
                <asp:RadioButton ID="RadioButtonWD2" GroupName="WD" runat="server" />
                <asp:Label ID="LabelWD3" runat="server" Text="| Количеством фотонов: "></asp:Label>
                <asp:RadioButton ID="RadioButtonWD3" GroupName="WD" runat="server" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell>
                &nbsp;
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Button ID="ButtonSaveDetectorsRanges" runat="server" Text="Сохранить информацию о пробегах (*.txt)"
                    OnClick="OnClickButtonSaveDetectorsRanges" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelDRI3" runat="server" Text="Целевые пробеги (с нормировкой): "></asp:Label>
                <asp:RadioButton ID="RadioButtonDRI3" GroupName="DRI" runat="server" />
                <asp:Label ID="LabelDRI4" runat="server" Text="| Целевые пробеги (без нормировки): "></asp:Label>
                <asp:RadioButton ID="RadioButtonDRI4" GroupName="DRI" runat="server" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelDRI1" runat="server" Text="Прочие пробеги (с нормировкой): "></asp:Label>
                <asp:RadioButton ID="RadioButtonDRI1" GroupName="DRI" runat="server" />
                <asp:Label ID="LabelDRI2" runat="server" Text="| Прочие пробеги (без нормировки): "></asp:Label>
                <asp:RadioButton ID="RadioButtonDRI2" GroupName="DRI" runat="server" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell>
                &nbsp;
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Button ID="ButtonSaveAll" runat="server" Text="Полная информация (*.txt)"
                    OnClick="OnClickButtonSaveAll" />
                <asp:Button ID="ButtonSaveTimeScales" runat="server" Text="Разбиение по времени (*.txt)"
                    OnClick="OnClickButtonSaveTimeScales"/>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell>
                &nbsp;
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelMode" runat="server" Text="Режим: "></asp:Label>
                <asp:DropDownList ID="DropDownListMode" runat="server" AutoPostBack="True">
                </asp:DropDownList>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelDetectorID" runat="server" Text="Номер детектора: "></asp:Label>
                <asp:DropDownList ID="DropDownListDetectorID" runat="server" AutoPostBack="True">
                </asp:DropDownList>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Right">
                <p>Прокрутка до:&#009;<a href="#XY">XY</a>,&#009;<a href="#XZ">XZ</a>,&#009;<a href="#YZ">YZ</a></p>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell>
                <hr />
            </asp:TableCell>
        </asp:TableRow>
         <asp:TableRow>
            <asp:TableCell>
                <a name="XY">XY:</a>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelXY_Img" runat="server"></asp:Label>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelXY_DDL" runat="server" Text="Z: "></asp:Label>
                <asp:DropDownList ID="DropDownListXY" Width="7%" runat="server" AutoPostBack="True">
                </asp:DropDownList>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Button ID="BUTTON_ORIGINALXY" runat="server" Text="Оригинал .jpg"
                    OnClick="OnClickButtonOriginal" />
                <asp:Button ID="BUTTON_MATRIXXY" runat="server" Text="Матрица .txt"
                    OnClick="OnClickButtonMatrix" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
        <asp:TableCell>
                <hr />
            </asp:TableCell>
        </asp:TableRow>
         <asp:TableRow>
            <asp:TableCell>
                <a name="XZ">XZ:</a>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelXZ_Img" runat="server"></asp:Label>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelXZ_DDL" runat="server" Text="Y: "></asp:Label>
                <asp:DropDownList ID="DropDownListXZ" Width="7%" runat="server" AutoPostBack="True">
                </asp:DropDownList>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Button ID="BUTTON_ORIGINALXZ" runat="server" Text="Оригинал .jpg"
                    OnClick="OnClickButtonOriginal" />
                <asp:Button ID="BUTTON_MATRIXXZ" runat="server" Text="Матрица .txt"
                    OnClick="OnClickButtonMatrix" />
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell>
                <hr />
            </asp:TableCell>
        </asp:TableRow>
         <asp:TableRow>
            <asp:TableCell>
                <a name="YZ">YZ:</a>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelYZ_Img" runat="server"></asp:Label>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Label ID="LabelYZ_DDL" runat="server" Text="X: "></asp:Label>
                <asp:DropDownList ID="DropDownListYZ" Width="7%" runat="server" AutoPostBack="True">
                </asp:DropDownList>
            </asp:TableCell>
        </asp:TableRow>
        <asp:TableRow>
            <asp:TableCell HorizontalAlign="Center">
                <asp:Button ID="BUTTON_ORIGINALYZ" runat="server" Text="Оригинал .jpg"
                    OnClick="OnClickButtonOriginal" />
                <asp:Button ID="BUTTON_MATRIXYZ" runat="server" Text="Матрица .txt"
                    OnClick="OnClickButtonMatrix" />
            </asp:TableCell>
        </asp:TableRow>
    </asp:Table>
    </div>
   
    </form>
</body>
</html>
