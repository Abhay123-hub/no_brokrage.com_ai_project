from typing import Annotated, TypedDict, Optional,Dict


def merge_dicts(existing: Optional[dict], new: dict) -> dict:
    if existing is None:
        return dict(new)
    existing.update(new)   # new wins on sub-key collisions
    return existing

class State(TypedDict):
    user_query:str
    output_dict:Dict
    df_dict:Annotated[dict, merge_dicts]
    response:str