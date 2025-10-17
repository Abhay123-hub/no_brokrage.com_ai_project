from typing import TypedDict,List,Dict,Optional,Any

class PropertyQuery(TypedDict, total=False):
    status: Optional[str]
    possessionDate: Optional[str]
    fullAddress: Optional[str]
    pincode: Optional[str]
    type: Optional[str]
    carpetArea: Optional[float]
    price: Optional[float]
    bathrooms: Optional[int]
    balcony: Optional[int]
    listingType: Optional[str]
    furnishedType: Optional[str]
    projectCategory: Optional[str]
    projectType: Optional[str]