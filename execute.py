from workflow import Workflow

app = Workflow()
inputs = {"user_query":"2 bhk furnished , in pune above 120 sqft upto price 2 crore should be ready before  dec 2025","df_dict":{}}
response = app.execute(inputs)
print(response)