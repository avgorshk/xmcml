<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="SURFACE.aspx.cs" Inherits="WEB_xmcml.SURFACE" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title>SURFACE-генератор</title>
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
    <asp:Table ID="TableSURFACE" runat="server" Width="75%" BorderColor="#000000" BorderWidth="1px">
        <asp:TableRow>
        <asp:TableCell>
            <asp:Label ID="LabelHeaderPublic" runat="server" Text="Генератор плоскостей:"></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelHeaderSurfaces" runat="server" Text="Количество плоскостей: "></asp:Label>
            <asp:DropDownList
                ID="DropDownListSurfaces" AutoPostBack="true" runat="server">
            </asp:DropDownList>
        </asp:TableCell>
    </asp:TableRow>
        </asp:Table>
    </div>

    </form>
</body>
</html>
