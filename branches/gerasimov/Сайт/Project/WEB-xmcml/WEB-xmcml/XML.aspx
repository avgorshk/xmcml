<%@ Page Language="C#" AutoEventWireup="true" CodeBehind="XML.aspx.cs" Inherits="WEB_xmcml.XML" %>

<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">

<html xmlns="http://www.w3.org/1999/xhtml">
<head runat="server">
    <title>XML-генератор</title>
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
    <asp:Table ID="TableXML" runat="server" Width="75%" BorderColor="#000000" BorderWidth="1px">
    <asp:TableRow>
        <asp:TableCell>
            <asp:Label ID="LabelHeaderBegin" runat="server" Text="Общее:"></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelNumberPhotons" runat="server" Text="Число фотонов: "></asp:Label>
            <asp:TextBox ID="TextBoxNumberPhotons" Text="100000" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelMinWeight" runat="server" Text=" Минимальный вес: "></asp:Label>
            <asp:TextBox ID="TextBoxMinWeight" Text="0.0000000001"  runat="server" MaxLength="25"></asp:TextBox>
        </asp:TableCell>
    </asp:TableRow>
     <asp:TableRow>
        <asp:TableCell>
            <asp:Label ID="LabelHeaderArea" runat="server" Text="Поле:"></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelHeaderCorner" runat="server" Text="Угол:"></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelCornerX" runat="server" Text="X: "></asp:Label>
            <asp:TextBox ID="TextBoxCornerX" Text="-10" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelCornerY" runat="server" Text=" Y: "></asp:Label>
            <asp:TextBox ID="TextBoxCornerY" Text="-10" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelCornerZ" runat="server" Text=" Z: "></asp:Label>
            <asp:TextBox ID="TextBoxCornerZ" Text="0" runat="server" MaxLength="25"></asp:TextBox>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelHeaderLength" runat="server" Text="Длина:"></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelLengthX" runat="server" Text="X: "></asp:Label>
            <asp:TextBox ID="TextBoxLengthX" Text="20" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelLengthY" runat="server" Text=" Y: "></asp:Label>
            <asp:TextBox ID="TextBoxLengthY" Text="20" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelLengthZ" runat="server" Text=" Z: "></asp:Label>
            <asp:TextBox ID="TextBoxLengthZ" Text="10" runat="server" MaxLength="25"></asp:TextBox>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelHeaderPartition" runat="server" Text="Разбиение:"></asp:Label>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelPartitionX" runat="server" Text="X: "></asp:Label>
            <asp:TextBox ID="TextBoxPartitionX" Text="40" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelPartitionY" runat="server" Text=" Y: "></asp:Label>
            <asp:TextBox ID="TextBoxPartitionY" Text="40" runat="server" MaxLength="25"></asp:TextBox>
            <asp:Label ID="LabelPartitionZ" runat="server" Text=" Z: "></asp:Label>
            <asp:TextBox ID="TextBoxPartitionZ" Text="20" runat="server" MaxLength="25"></asp:TextBox>
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell>
            &nbsp;
        </asp:TableCell>
    </asp:TableRow>
    <asp:TableRow>
        <asp:TableCell HorizontalAlign="Center">
            <asp:Label ID="LabelHeaderLayers" runat="server" Text="Количество слоев: "></asp:Label>
            <asp:DropDownList
                ID="DropDownListLayers" AutoPostBack="true" runat="server">
            </asp:DropDownList>
        </asp:TableCell>
    </asp:TableRow>
        </asp:Table>
    </div>
    </form>
</body>
</html>
