import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langgraph.graph import START, END,StateGraph
from typing import TypedDict,List,Dict,Optional,Any
from llm_manager import LLM
from datetime import datetime, date
import re
import json

get_LLM = LLM()
from State import State
df = pd.read_csv("cleaned_master_for_chatbot.csv")

llm = get_LLM.get_llm() ## simple llm
structured_llm = get_LLM.get_structured_llm() ## structured llm 

class Agent:
    def __init__(self):
        pass


    def main_agent(self,state):
        """
    Takes a user query like:
        "Show me 3BHK flats in Pune under ₹1.2 Cr"
    Returns a dictionary:
        {
          "type": "3BHK",
          "fullAddress": "Pune",
          "price": 12000000
        }
    """

        template = """
        You are a strict JSON extractor for real-estate search queries.

        Task:
        Given a user's natural-language real-estate query, output **ONLY** a single valid JSON object (no extra text, no explanation). The JSON object must include ZERO or more keys from this exact set:

        [
        "status",
        "possessionDate",
        "fullAddress",
        "pincode",
        "type",
        "carpetArea",
        "price",
        "bathrooms",
        "balcony",
        "listingType",
        "furnishedType",
        "projectCategory",
        "projectType"
        ]

        Output rules & normalization:
        1. Price:
        - Convert prices to numeric **rupees** (integers or floats).  
            *Examples:* "1.2 Cr" -> 12000000, "80 lakh" -> 8000000.
        
        - If user says **"above"/"over"**, set `"price"` to a **single numeric value** representing the lower bound.
            *Example:* "above 50 lakh" -> `"price": 5000000`
        - If user gives a **range** (`"between X and Y"`), set `"price"` to an object with `{{"min": <rupees>, "max": <rupees>}}.
            *Example:* "between 50 and 80 lakh" -> `"price": {{"min": 5000000, "max": 8000000}}
        - - If user says **"under"/"below"/"up to"**, set `"price"` to a **single numeric value** representing the upper bound.
            *Example:* "under ₹1.2 Cr" -> `{{"min":0,"max":12000000}}
            IMPORTANT: price output format rules (BE VERY STRICT)
        - If the user expresses an upper bound using words like "under", "below", "up to", or "maximum", the model MUST output the price as an object with "min" and "max". Use min = 0 and max = <rupees value>.
        Example: "under 1.2 Cr"  ->  "price": {{"min": 0, "max": 12000000}}
        - If the user expresses a lower bound using words like "above", "over", "minimum", the model MUST output price as an object with "min" and "max". Use max = null (or omit max) and min = <rupees value>.
        Example: "above 50 lakh" -> "price": {{"min": 5000000, "max": null}}
        - If the user gives a closed range ("between X and Y"), the model MUST output price as {{"min": <rupees>, "max": <rupees>}}.
        - If the user gives an exact price (e.g., "price 12,000,000" or "budget 1.2 Cr" with no comparison word), the model MAY output either a numeric value or an object; prefer object for consistency is allowed but not required.
        - Always convert human units to rupees (1.2 Cr -> 12000000, 80 lakh -> 8000000).


        2. Type (BHK):
        - Use the format `"1BHK"`, `"2BHK"`, `"3BHK"`, etc. Normalize words ("three bhk") to numeric form.

        3. possessionDate:
        - This represents the **expected date when the property will be ready for possession (handover date)**.
        - If the property is already ready to move in, use the status "READY_TO_MOVE" instead of setting a future possessionDate.
        - If the user specifies a **future date, year, or range**, convert it into the ISO date format "YYYY-MM-DD".
        - If the user only mentions a **year**, assume the start of that year ("YYYY-01-01").
            Example: "possession in 2025" → "possessionDate": "2025-01-01"
        - If the user mentions **by, before, or until a year**, treat it as **upper bound**.
            Example: "possession by 2026" → "possessionDate": "2026-01-01"
        - If the user says **after or from a year**, treat it as **lower bound**.
            Example: "possession after 2024" → "possessionDate": "2024-01-01"
        - If the user says **ready to move / already ready / immediate possession**, do **NOT** set possessionDate.
            Instead, set `"status": "READY_TO_MOVE"`.
        - Always use a valid date string in ISO format. Examples:
            "2025" → "2025-01-01"
            "Dec 2024" → "2024-12-01"
            "after 2023" → "2023-01-01"
        - If no clear possession information is given, **omit the key** (do not guess or hallucinate).
        - Do not output explanatory text — only the JSON key-value pair.

        Examples (strictly follow these):

        Example 1:
        User: "Show me 2BHK flats ready in 2025"
        Output:
        {{"type": "2BHK", "possessionDate": "2025-01-01"}}

        Example 2:
        User: "Flats with possession by 2026 in Mumbai"
        Output:
        {{"fullAddress": "Mumbai", "possessionDate": "2026-01-01"}}

        Example 3:
        User: "3BHK ready to move apartments in Pune"
        Output:
        {{"type": "3BHK", "status": "READY_TO_MOVE", "fullAddress": "Pune"}}

        Example 4:
        User: "Possession after 2024 near Baner"
        Output:
        {{"possessionDate": "2024-01-01", "fullAddress": "Baner"}}

        Example 5:
        User: "Ready to move property"
        Output:
        {{"status": "READY_TO_MOVE"}}

        Example 6:
        User: "Expected possession in December 2025"
        Output:
        {{"possessionDate": "2025-12-01"}}

        4. fullAddress:
        - Prefer the city / locality string that appears in the query (capitalized). e.g., "Pune", "Wakad, Pune", "Chembur, Mumbai".
        - If the user gives a pincode, include `"pincode"` (string) too.

        5. pincode:
        - 6-digit postal code as string.

        6. carpetArea:
        - Numeric area in square feet (integer or float). Accept inputs like "800 sqft" -> 800.

        7. bathrooms and balcony:
        - Integers.

        8. listingType:
        - Normalize to `"Sell"` (for sale/resale) or `"Rent"`.

        9. furnishedType:
        - Normalize to `"FURNISHED"`, `"SEMI_FURNISHED"`, or `"UNFURNISHED"`.

        10. projectCategory:
            - `"STANDALONE"` or `"COMPLEX"`.

        11. projectType (and propertyCategory if implied):
            - `"RESIDENTIAL"` or `"COMMERCIAL"`.
        12. status (The current status of the project):
            - 'UNDER_CONSTRUCTION'or  'READY_TO_MOVE'


        12. If a key isn't clearly mentioned or implied in the query, DO NOT include that key.

        13. Always output valid JSON only. Do not output explanatory text, markdown, or code fences.

        14. If the user provides conflicting specifications, choose the most recent explicit statement in the query. If still ambiguous, omit the conflicting key.

        Examples (these are strict examples you must follow):

        Example A:
        User: "Show me 3BHK flats in Pune of ₹1.2 Cr"
        Output:
        {{"type":"3BHK","location":"Pune","price": 12000000 }}

        Example B:
        User: "2 bhk semi-furnished for rent in Wakad, Pune above 800 sqft"
        Output:
        {{"type":"2BHK","listingType":"Rent","furnishedType":"SEMI_FURNISHED","fullAddress":"Wakad, Pune","carpetArea":800}}

        Example C:
        User: "Any ready-to-move 1 BHK in Mumbai pincode 400075"
        Output:
        {{"type":"1BHK","status":"READY_TO_MOVE","fullAddress":"Mumbai","pincode":"400075"}}

        Example D:
        User: "3BHK with 2 bathrooms and 2 balconies in Bangalore under 90 lakh"
        Output:
        {{"type":"3BHK","bathrooms":2,"balcony":2,"fullAddress":"Bangalore","price":9000000}}

        Example E:
        User: "Looking for a standalone complex, 4BHK, possession by 2026, fully furnished"
        Output:
        {{"projectCategory":"STANDALONE","type":"4BHK","possessionDate":"2026-01-01","furnishedType":"FURNISHED"}}



        Example F:
        User: "I want a commercial shop for rent near MG Road, Bangalore, budget 1.5 Cr"
        Output:
        {{"projectType":"COMMERCIAL","listingType":"Rent","fullAddress":"MG Road, Bangalore","price":15000000}}

        Example G:
        User: "Unfurnished 1 BHK for sale, carpet area 450 sqft, under 35 L"
        Output:
        {{"type":"1BHK","furnishedType":"UNFURNISHED","listingType":"Sell","carpetArea":450,"price":3500000}}

        Example H:
        User: "Flats ready in 2024 in Chembur with at least 2 balconies"
        Output:
        {{"possessionDate":"2024-01-01","fullAddress":"Chembur","balcony":2}}

        Example I:
        User: "2 BHK resale in Noida with 2 bathrooms and pincode 201301"
        Output:
        {{"type":"2BHK","listingType":"Sell","fullAddress":"Noida","bathrooms":2,"pincode":"201301"}}

        Now, process this user query and **output ONLY** the JSON object (no commentary):

        User: {user_query}
            
            

                                                                                        """
        user_query = state.get("user_query")
        prompt = PromptTemplate(template = template,input_variables = ["user_query"])

        chain = prompt | structured_llm 
        response = chain.invoke({"user_query":user_query})
        return {"output_dict":response}
    
    def status_agent(self,state:State):
         ## first of all i will fetching the output_dict from the state
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "status" in list(output_dict.keys()):
            status = output_dict.get("status")
            filtered_df = df[df["status"].astype(str).str.upper() == status.strip().upper()].copy()
            df_dict["filtered_df_status"] = filtered_df 
        else:
            df_dict["filtered_df_status"] = None
        
        return {"df_dict":df_dict}
    
    def furnished_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "furnishedType" in list(output_dict.keys()):
            furnishedType = output_dict.get("furnishedType")
            filtered_df = df[df["furnishedType"].astype(str).str.upper() == furnishedType.strip().upper()].copy()
            df_dict["filtered_df_furnished"] = filtered_df 
        else:
            df_dict["filtered_df_status"] = None
        return {"df_dict":df_dict} 
    
    def type_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "type" in list(output_dict.keys()):
            type = output_dict.get("type")
            filtered_df = df[df["type"].astype(str).str.upper() == type.strip().upper()].copy()
            df_dict["filtered_df_type"] = filtered_df 
        else:
            df_dict["filtered_df_type"] = None
        return {"df_dict":df_dict}
    
    def listingType_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "listingType" in list(output_dict.keys()):
            listing = output_dict.get("listingType")
            filtered_df = df[df["listingType"] == listing].copy()
            df_dict["filtered_df_listing"] = filtered_df
        else:
            df_dict["filtered_df_listing"] = None
        return {"df_dict":df_dict}
    
    def carpet_area_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "carpetArea" in list(output_dict.keys()):
            area = output_dict.get("carpetArea")
            filtered_df = df[df["carpetArea"] == area].copy()
            df_dict["filtered_df_area"] = filtered_df
        else:
            df_dict["filtered_df_area"] = None
        return {"df_dict":df_dict}
    
    def price_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")

        if "price" in list(output_dict.keys()):
            price = output_dict.get("price")
            if isinstance(price,dict):
                min_price = price.get("min") ## minimum price
                max_price = price.get("max") ## maximum price
                filtered_df = df[(df['price'] > min_price) & (df['price'] < max_price)].copy()
            elif isinstance(price,int):
                filtered_df = df[df["price"] == price].copy()
            else:
                filtered_df = None
            df_dict["filtered_df_price"] = filtered_df
        else:
            df_dict["filtered_df_price"] = None
        return {"df_dict":df_dict}
    
    def convert_to_date(self,date_input):
        """
        Converts any date-like input (string, int year, datetime, etc.)
        into a Python `date` object (YYYY-MM-DD) with no time part.
        
        Works great for comparisons and filtering in pandas.

        Examples:
            'Dec 2025'     -> date(2025, 12, 1)
            '2025-12-25'   -> date(2025, 12, 25)
            '2025'         -> date(2025, 1, 1)
            datetime(2024,3,5,9,0) -> date(2024, 3, 5)
            None or invalid -> None
        """
        if date_input is None or (isinstance(date_input, str) and not date_input.strip()):
            return None

        # Already a datetime or date
        if isinstance(date_input, datetime):
            return date_input.date()
        if isinstance(date_input, date):
            return date_input  # already date object

        # If it's a year (int or string)
        try:
            year = int(date_input)
            if 1900 <= year <= 2100:
                return date(year, 1, 1)
        except Exception:
            pass

        # ✅ Correct parsing for ISO formats (YYYY-MM-DD)
        try:
            dt = pd.to_datetime(str(date_input), errors="coerce")
            if pd.notna(dt):
                return dt.date()  # convert to date (no time)
        except Exception:
            pass

        # Fallback: find a 4-digit year in text
        match = re.search(r"(20\d{2})", str(date_input))
        if match:
            return date(int(match.group(1)), 1, 1)

        return None
        

    def possession_date_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "possessionDate" in list(output_dict.keys()):
            date = output_dict.get("possessionDate")
            date_object = self.convert_to_date(date)
            df_test = df.copy()
            df_test["possessionDate"] = df_test["possessionDate"].map(lambda x: str(x).split(" ")[0] if pd.notna(x) else None)

            df_test["possessionDate"] = df_test["possessionDate"].map(self.convert_to_date)
            #  filter rows where possessionDate is not null and < target_date
            filtered_df = df_test[df_test["possessionDate"].notna() & (df_test["possessionDate"] < date_object)].copy()
            df_dict["filtered_df_possession"] = filtered_df
        else:
            df_dict["filtered_df_possession"] = None
        return {"df_dict":df_dict}
    
    def bathroom_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "bathrooms" in list(output_dict.keys()):
            bathrooms = output_dict.get("bathrooms")
            filtered_df = df[df["bathrooms"] == bathrooms].copy()
            df_dict["filtered_df_bathrooms"] = filtered_df
        else:
            df_dict["filtered_df_bathrooms"] = None
        return {"df_dict":df_dict}
    
    def balcony_agent(self,state:State):
        output_dict = state.get("output_dict")
        df_dict = state.get("df_dict")
        if "balcony" in list(output_dict.keys()):
            balcony = output_dict.get("balcony")
            filtered_df = df[df["balcony"] == balcony].copy()
            df_dict["filtered_df_balcony"] = filtered_df
        else:
            df_dict["filtered_df_balcony"] = None
        return {"df_dict":df_dict}
    
    def retrieve_agent(self,state: State):
        output_dict = state.get("output_dict", {})
        df_dict = state.get("df_dict", {})

        list_agents = list(output_dict.keys())
        list_df = []

        # --- collect all filtered dataframes that exist ---
        if "status" in list_agents and df_dict.get("filtered_df_status") is not None:
            list_df.append(df_dict["filtered_df_status"])

        if "furnishedType" in list_agents and df_dict.get("filtered_df_furnished") is not None:
            list_df.append(df_dict["filtered_df_furnished"])

        if "type" in list_agents and df_dict.get("filtered_df_type") is not None:
            list_df.append(df_dict["filtered_df_type"])

        if "listingType" in list_agents and df_dict.get("filtered_df_listing") is not None:
            list_df.append(df_dict["filtered_df_listing"])

        if "carpetArea" in list_agents and df_dict.get("filtered_df_area") is not None:
            list_df.append(df_dict["filtered_df_area"])

        if "price" in list_agents and df_dict.get("filtered_df_price") is not None:
            list_df.append(df_dict["filtered_df_price"])

        if "possessionDate" in list_agents and df_dict.get("filtered_df_possession") is not None:
            list_df.append(df_dict["filtered_df_possession"])

        if "bathrooms" in list_agents and df_dict.get("filtered_df_bathrooms") is not None:
            list_df.append(df_dict["filtered_df_bathrooms"])

        if "balcony" in list_agents and df_dict.get("filtered_df_balcony") is not None:
            list_df.append(df_dict["filtered_df_balcony"])

        # --- Step 2: find intersection of all filtered DataFrames ---
        if not list_df:
            # if no filters were applied
            final_df = None
        else:
            # start with the first dataframe
            final_df = list_df[0]
            # iteratively keep only common rows
            for df_part in list_df[1:]:
                final_df = pd.merge(final_df, df_part, how="inner")

        # --- Step 3: store the final filtered dataframe ---
        df_dict["final_filtered_df"] = final_df
        state["df_dict"] = df_dict

        return {"df_dict":df_dict}
    
    def final_agent(self,state:State):
        df_dict = state.get("df_dict")
        output_dict = state.get("output_dict")
        final_df = df_dict.get("final_filtered_df")
        final_df_list = final_df.to_dict(orient='records')

        user_query = state.get("user_query")
        template = """ You are an intelligent system.You will be given final dataset in list format {final_df_list}
        along with output dictionary provided in a dictionary  {output_dict}.These are the columns of the dataset 
        ['projectName', 'status', 'possessionDate', 'fullAddress', 'pincode',
        'type', 'carpetArea', 'price', 'bathrooms', 'balcony', 'listingType',
        'furnishedType', 'projectCategory', 'projectType', 'propertyCategory',
        'landmark', 'cityId', 'localityId', 'subLocalityId', 'propertyImages',
        'floorPlanImage'].
        and this is the orignal user query {user_query}.
        based on this following data give a final response.This final response will be holding the answer of the query
        based on the data of final dataset and output dictionary.The response should be in simple langauge covering
        all the information about the user query what every you think the user is expecting for.Before generating the query think two three times
        while taking the reference of provided data.

        ImportantNote:  While generating the response keep it like human to human conversation without missing any important details
        ,it should be like you are giving all details to the user but user should not feel that he/she is communicating with the machine
        

                        """
        

        prompt = PromptTemplate(template = template,input_variables = ["final_df_dict","output_dict","user_query"])
        chain = prompt | llm 
        response = chain.invoke({"final_df_list":final_df_list,"output_dict":json.dumps(output_dict),"user_query":user_query})
        return {"response":response}
