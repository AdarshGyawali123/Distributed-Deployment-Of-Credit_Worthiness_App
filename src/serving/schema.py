from typing import Dict, Tuple,Type
from pydantic import BaseModel, create_model
from mlflow.types.schema import DataType
from mlflow.models.signature import ModelSignature




def mlflow_datatype_check(type:DataType,name:str):
    if type == DataType.boolean:
        return bool
    if type in(DataType.float,DataType.double):
        return float
    if type in (DataType.long,DataType.integer):
        return int
    if type == (DataType.string):
        return str
    
    raise ValueError(f"Unsupported MLflow type: {type}- {name}")


def build_request_model_from_signature(signature: ModelSignature):
    fields : Dict[str,Tuple[type,...]]={}

    for col in signature.inputs.inputs:
        name = col.name
        mlflow_check = mlflow_datatype_check(col.type,col.name)

        fields[name]=(mlflow_check,...)

    OurModel = create_model("OurModel",**fields,)

    return OurModel

        
        





    



# def mlflow_type_to_python(dtype: DataType):
#     if dtype == DataType.boolean:
#         return bool
#     if dtype in (DataType.integer, DataType.long):
#         return int
#     if dtype in (DataType.float, DataType.double):
#         return float
#     if dtype == DataType.string:
#         return str
#     raise ValueError(f"Unsupported MLflow type: {dtype}")

# def build_request_model_from_signature(
#     signature: ModelSignature,
# ) -> Type[BaseModel]:
#     """
#     Dynamically create a strict Pydantic model
#     from the MLflow input signature.
#     """
#     if signature is None or signature.inputs is None:
#         raise RuntimeError("Model signature inputs are missing")

#     fields: Dict[str, Tuple[type, ...]] = {}

#     for col in signature.inputs.inputs:
#         name = col.name
#         py_type = mlflow_type_to_python(col.type)

#         # REQUIRED field → no default
#         fields[name] = (py_type, ...)

#     RequestModel = create_model(
#         "InferenceRequest",
#         **fields,
#     )

#     # STRICT: reject extra fields
#     class Config:
#         extra = "forbid"

#     RequestModel.Config = Config

#     return RequestModel


