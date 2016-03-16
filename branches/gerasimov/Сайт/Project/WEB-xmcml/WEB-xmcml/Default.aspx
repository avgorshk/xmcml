<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="Default.aspx.cs" Inherits="WEB_xmcml._Default" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title>Главная страница</title>
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
    <asp:Table ID="TableMain" runat="server" Width="75%" BorderColor="#000000" BorderWidth="1px" >
    <asp:TableRow>
        <asp:TableCell>
                <asp:Label ID="LabelXML" runat="server" Text="XML: "></asp:Label>
                <asp:FileUpload ID="FileUploadXML" runat="server" ToolTip="Выберите XML файл для загрузки" />
        </asp:TableCell>
        <asp:TableCell>
                <asp:Label ID="LabelSURFACE" runat="server" Text="SURFACE: "></asp:Label>
                <asp:FileUpload ID="FileUploadSURFACE" runat="server" ToolTip="Выберите SURFACE файл для загрузки" />
        </asp:TableCell>
    </asp:TableRow>
        <asp:TableRow>
        <asp:TableCell ColumnSpan="2" HorizontalAlign="Center">
                <asp:Label ID="LabelNameTasks" runat="server" Text="Имя задачи: " ToolTip="Имя задачи которое будет отображаться в таблице"></asp:Label>
                <asp:TextBox ID="TextBoxNameTasks" runat="server" MaxLength="25"></asp:TextBox>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell ColumnSpan="2" HorizontalAlign="Center">
                <asp:Label ID="LabelError" runat="server" ForeColor="#FF0000" Text=""></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell ColumnSpan="2" HorizontalAlign="Center">
                <asp:Button ID="ButtonSend" runat="server" Text="Добавить задачу" OnClick="ButtonSend_Click"/>
        </asp:TableCell>
    </asp:TableRow>
    </asp:Table>
    </div>
    </form>
</body>
</html>
