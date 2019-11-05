from mycbrwrapper.rest import getRequest
from mycbrwrapper.concepts import Concept, Concepts
from dataset.dataset import *
from dataset.dataset_to_sklearn import fromDataSetToSKLearn
import json

__name__ = "dataset_to_cbr"

defaulthost = "localhost:8080"
def getDoubleAttributeParamterJSON(min,max,solution):
    return """
    {{
    "type": "Double",
    "min": "{}",
    "max": "{}",
    "solution": "{}"
    }}
    """.format(min,max,solution)

def getStringAttributeParamterJSON(solution):
    return """
    {{
    "type": "String",
    "solution": "{}"
    }}
    """.format(solution)
def getStringAttributeParamterJSON(allowedvalues,solution):
    return """
    {{
    "type": "Symbol",
    "allowedValues": [{}],
    "solution": "{}"
    }}
    """.format(allowedvalues,solution)
def getDoubleParameters(imin,imax,solution):
    return {"attributeJSON":getDoubleAttributeParamterJSON(imin,imax,solution)}

def getStringParameters(solution):
    return {"attributeJSON":getStringAttributeParamterJSON(solution)}

def findColFromValue(colmap, value):
    for key,dict in colmap.items():
        if "possible_values" in dict and value in dict["possible_values"]:
            return key
    return None
def findDatasetInfo(datasetInfo, name):
    for row in datasetInfo["cols"]:
        if row["name"] is name:
            return row
    return None

def fromDatasetToCBR(dataset,sklearndataset,colmap):
    #print(dataset.df)

    #sklearndataset, colmap = fromDataSetToSKLearn(dataset)

    datadf = sklearndataset.getDataFrame()
    sklearncols = list(datadf)


    cs = Concepts(defaulthost)
    c = cs.addConcept(dataset.name)
    # create the model in the CBR system
    # for col in dataset.getTypes():
    #     #print("columns \"{}\"".format(dataset.df.columns))
    #     colname = col["name"]
    #     #print("coltype: {}".format(col["type"]))
    #     if colname in nominalcols:
    #
    #     elif (col["type"] is "str" or "nominal") and (isinstance(col["type"],str)):
    #         #print("creating new str attribute from coltype: {}".format(col["type"]))
    #         c.addAttribute(colname,getStringParameters())
    #     else:
    #         #print("creating new double attribute from coltype: {}".format(col["type"]))
    #         cmin = dataset.getMinForCol(colname)
    #         cmax = dataset.getMaxForCol(colname)
    #         c.addAttribute(colname,
    #                        getDoubleParameters(cmin, cmax))
    # for colname,value_list in colmap.items():
    #     #print("columns \"{}\"".format(dataset.df.columns))
    #     #print("coltype: {}".format(col["type"]))
    #
    #     if value_list["type"] is  "nominal":
    #         #print("creating new str attribute from coltype: {}".format(col["type"]))
    #         c.addAttribute(colname,getStringParameters())
    #     else:
    #         #print("creating new double attribute from coltype: {}".format(col["type"]))
    #         cmin = dataset.getMinForCol(colname)
    #         cmax = dataset.getMaxForCol(colname)
    #         c.addAttribute(colname,
    #                        getDoubleParameters(cmin, cmax))
    for col in sklearncols:
        if col not in colmap: #it has to be a binarized new-column
            #print(f"sending paramstring: {paramstr}")
            originalColName = findColFromValue(colmap,col)
            datasetInfoRow = findDatasetInfo(dataset.datasetInfo, originalColName)
            classCol = datasetInfoRow["class"]
            paramstr = getDoubleAttributeParamterJSON(0, 1, classCol)
            c.addAttribute(col,paramstr)
        elif colmap[col]["type"] is "number":
              #cmin = dataset.getMinForCol(col)
              #cmax = dataset.getMaxForCol(col)
              datasetInfoRow = findDatasetInfo(dataset.datasetInfo, col)
              classCol = datasetInfoRow["class"]
              c.addAttribute(col,
                             getDoubleAttributeParamterJSON(0, 1.0, classCol))
        elif col in colmap and colmap[col]["type"] is "nominal":
            cmin = datadf[col].min()
            cmax = datadf[col].max()
            datasetInfoRow = findDatasetInfo(dataset.datasetInfo, col)
            classCol = datasetInfoRow["class"]
            paramstr = getDoubleAttributeParamterJSON(cmin, cmax, classCol)
            c.addAttribute(col, paramstr)



    # create the instances that fit into the model
    prefix = c.name
    jsonstr = sklearndataset.getDataFrame().to_json(orient="records")
    jsono = json.loads(jsonstr)
    counter = 0
    # for row in jsono:
    #     counter += 1
    #     row["caseID"] = f"{prefix}{counter}"
    datadict = {}
    # this can later be used to mass add instances
    datadict["cases"] = jsono
    c.addCaseBase("mydefaultCB")
    # for case in jsono:
    #     //onlycase = case.copy()
    #     //onlycase.pop("caseID", None)
    #     tempdict = {}
    #
    #
    #     tempdict["case"] = case
    #    c.addInstance(case["caseID"],json.dumps(case),"mydefaultCB")
    c.addInstances(datadict, "mydefaultCB")
    return cs, c

