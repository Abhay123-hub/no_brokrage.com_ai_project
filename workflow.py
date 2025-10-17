from langgraph.graph import StateGraph, END,START
from Agents import Agent

from State import State
class Workflow:
    def __init__(self):
        self.agent = Agent()
    
    def create_workflow(self):
        workflow = StateGraph(State)
        ## defining all the nodes
        workflow.add_node("main_agent",self.agent.main_agent)
        workflow.add_node("status_agent",self.agent.status_agent)
        workflow.add_node("furnished_agent",self.agent.furnished_agent)
        workflow.add_node("type_agent",self.agent.type_agent)
        workflow.add_node("listingType_agent",self.agent.listingType_agent)
        workflow.add_node("carpet_area_agent",self.agent.carpet_area_agent)
        workflow.add_node("price_agent",self.agent.price_agent)
        workflow.add_node("possession_date_agent",self.agent.possession_date_agent)
        workflow.add_node("bathroom_agent",self.agent.bathroom_agent)
        workflow.add_node("balcony_agent",self.agent.balcony_agent)
        workflow.add_node("retrieve_agent",self.agent.retrieve_agent)
        workflow.add_node("final_agent",self.agent.final_agent)

        ## connecting the nodes with each other
        workflow.add_edge(START,"main_agent")
        workflow.add_edge("main_agent","status_agent")
        workflow.add_edge("main_agent","furnished_agent")
        workflow.add_edge("main_agent","type_agent")
        workflow.add_edge("main_agent","listingType_agent")
        workflow.add_edge("main_agent","carpet_area_agent")
        workflow.add_edge("main_agent","price_agent")
        workflow.add_edge("main_agent","possession_date_agent")
        workflow.add_edge("main_agent","bathroom_agent")
        workflow.add_edge("main_agent","balcony_agent")

        workflow.add_edge("status_agent","retrieve_agent")
        workflow.add_edge("furnished_agent","retrieve_agent")
        workflow.add_edge("type_agent","retrieve_agent")
        workflow.add_edge("listingType_agent","retrieve_agent")
        workflow.add_edge("carpet_area_agent","retrieve_agent")
        workflow.add_edge("price_agent","retrieve_agent")
        workflow.add_edge("possession_date_agent","retrieve_agent")
        workflow.add_edge("bathroom_agent","retrieve_agent")
        workflow.add_edge("balcony_agent","retrieve_agent")
        workflow.add_edge("retrieve_agent","final_agent")
        workflow.add_edge("final_agent",END)

        graph = workflow.compile()
        return graph
    
    def execute(self,inputs):
        graph = self.create_workflow()

        try:
            response = graph.invoke(inputs)
            return response["response"].content
        except Exception as e:
            raise e 

