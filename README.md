# 🏙️ Real Estate Agentic AI Chatbot

An **intelligent real-estate assistant** built using **LangChain**, **LangGraph**, and **Streamlit** that allows users to query property listings in natural language —  
for example:
> “Show me 3 BHK flats in Pune under ₹1.2 Cr that are ready to move.”

The system breaks the query into structured filters using multiple specialized agents (like `price_agent`, `status_agent`, `type_agent`), filters the dataset, and returns human-friendly answers with property images.

---

## 🧠 Features

- **Agentic AI Framework:**  
  Designed using LangGraph where a **main agent** parses user queries into structured key–value filters, and **sub-agents** handle each column such as price, type, status, furnishedType, etc.

- **Parallel Agent Workflow:**  
  Each agent runs in parallel, filtering data for its column. The final retriever agent merges results to produce a single filtered dataset.

- **LLM-powered Query Understanding:**  
  Built using **LangChain** for LLM calling, prompting, and chaining. The main agent converts user queries into a structured dictionary using prompt engineering.

- **Data Cleaning & Pre-processing:**  
  Combined multiple CSV files (`Project.csv`, `ProjectConfiguration.csv`, `ProjectConfigurationVariant.csv`, etc.) into a unified, cleaned dataset of 21 columns.

- **Visual Output:**  
  Used **Streamlit** to build a simple and interactive frontend that displays final results, property details, and images fetched from URLs in the dataset.

- **Human-like Answers:**  
  The final agent generates simple, easy-to-understand summaries about the properties matching the user’s query.

---

## 🧰 Tech Stack

| Component | Technology Used |
|------------|------------------|
| **Language** | Python 3.10 |
| **LLM Framework** | [LangChain](https://www.langchain.com/) |
| **Agent Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) |
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Data Handling** | Pandas |
| **LLM Provider** | OpenAI (GPT-4o-mini / GPT-3.5-turbo) |
| **Visualization** | Streamlit UI with Markdown and image rendering |

---

## 🧩 Project Workflow

1. **Data Collection & Cleaning**
   - Loaded and merged multiple project-related CSV files.
   - Cleaned and standardized columns such as `price`, `possessionDate`, `listingType`, etc.

2. **Main Agent (Query Parser)**
   - Converts natural-language queries into structured JSON using LangChain prompt templates.  
     Example:  
     ```json
     {
       "type": "3BHK",
       "fullAddress": "Pune",
       "price": {"min": 0, "max": 12000000}
     }
     ```

3. **Sub-Agents (Column-based Filters)**
   - Each agent filters the dataset for a specific attribute:
     - `price_agent` → filters price range  
     - `type_agent` → filters BHK type  
     - `status_agent` → filters project status  
     - `furnished_agent`, `balcony_agent`, etc.

4. **Retrieve Agent**
   - Intersects all filtered DataFrames to produce a final filtered result containing only rows that satisfy all user conditions.

5. **Final Agent**
   - Generates a natural-language response with property summaries and image URLs.
   - Output example:
     ```
     Found 3 properties matching your criteria.
     1. Ashwini | 3 BHK | ₹1.2 Cr | 1200 sqft | Ready to move | Wakad, Pune
     ```

6. **Streamlit App**
   - Provides an interactive chat interface.
   - Users can input queries and instantly see summarized property results with images.

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/real-estate-agentic-ai.git
   cd real-estate-agentic-ai
