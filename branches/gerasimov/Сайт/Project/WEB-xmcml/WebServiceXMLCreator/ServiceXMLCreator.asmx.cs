using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Services;

namespace WebServiceXMLCreator
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

        [WebMethod]
        public String GetXML(String numberOfPhotons, String minWeight, String[] corner,
            String[] length, String[] partition, int numLayers, String[] refractiveIndex,
                String[] absorptionCoefficient, String[] scatteringCoefficient, String[] anisotropy)
        {
            String XMLFile = "";

            XMLFile += "<?xml version=\"1.0\"?>\r\n<Input>\r\n";
            XMLFile += "\t<NumberOfPhotons>" + numberOfPhotons + "</NumberOfPhotons>\r\n";
            XMLFile += "\t<MinWeight>" + minWeight + "</MinWeight>\r\n";
            
            XMLFile += "\t<Area>\r\n";
            
            XMLFile += "\t\t<Corner>\r\n\t\t\t<X>" + corner[0] + "</X>\r\n"
                + "\t\t\t<Y>" + corner[1] + "</Y>\r\n" + "\t\t\t<Z>" + corner[2] + "</Z>\r\n\t\t</Corner>\r\n";
            
            XMLFile += "\t\t<Length>\r\n\t\t\t<X>" + length[0] + "</X>\r\n"
                + "\t\t\t<Y>" + length[1] + "</Y>\r\n" + "\t\t\t<Z>" + length[2] + "</Z>\r\n\t\t</Length>\r\n";

            XMLFile += "\t\t<PartitionNumber>\r\n\t\t\t<X>" + partition[0] + "</X>\r\n"
                + "\t\t\t<Y>" + partition[1] + "</Y>\r\n" + "\t\t\t<Z>" + partition[2] + "</Z>\r\n\t\t</PartitionNumber>\r\n";

            XMLFile += "\t</Area>\r\n";

            XMLFile += "\t<NumberOfLayers>" + numLayers.ToString() + "</NumberOfLayers>\r\n";

            XMLFile += "\t<Layers>\r\n";

            for (int i = 0; i < numLayers; i++)
            {
                XMLFile += "\t\t<Layer>\r\n";

                XMLFile += "\t\t\t<RefractiveIndex>" + refractiveIndex[i] + "</RefractiveIndex>\r\n";
                XMLFile += "\t\t\t<AbsorptionCoefficient>" + absorptionCoefficient[i] + "</AbsorptionCoefficient>\r\n";
                XMLFile += "\t\t\t<ScatteringCoefficient>" + scatteringCoefficient[i] + "</ScatteringCoefficient>\r\n";
                XMLFile += "\t\t\t<Anisotropy>" + anisotropy[i] + "</Anisotropy>\r\n";

                XMLFile += "\t\t\t<NumberOfSurfaces>" + numLayers.ToString() + "</NumberOfSurfaces>\r\n";

                XMLFile += "\t\t\t<SurfaceIds>\r\n";

                for (int j = 0; j < numLayers; j++)
                {
                    XMLFile += "\t\t\t\t<Id>" + j.ToString() + "</Id>\r\n";
                }

                XMLFile += "\t\t\t</SurfaceIds>\r\n";

                XMLFile += "\t\t</Layer>\r\n";
            }

            XMLFile += "\t</Layers>\r\n";

            XMLFile += "\t<NumberOfSurfaces>" + numLayers.ToString() + "</NumberOfSurfaces>\r\n";
            XMLFile += "</Input>";

            return XMLFile;
        }
    }
}