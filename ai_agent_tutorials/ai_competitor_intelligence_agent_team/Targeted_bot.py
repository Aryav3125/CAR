import streamlit as st

from agno.tools.firecrawl import FirecrawlTools
import pandas as pd
import requests
from firecrawl import FirecrawlApp
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

if "groq_api_key" in st.session_state:
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
# Streamlit UI
st.set_page_config(page_title="Competitor Analysis Agent", layout="wide")

# Sidebar for API keys
st.sidebar.title("API Keys")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")
tavily_api_key = st.sidebar.text_input("Tavily API Key", type="password")

# # Add search engine selection before API keys
# search_engine = st.sidebar.selectbox(
#     "Select Search Endpoint",
#     options=["Perplexity AI - Sonar Pro", "Exa AI"],
#     help="Choose which AI service to use for finding competitor URLs"
# )
#
# # Show relevant API key input based on selection
# if search_engine == "Perplexity AI - Sonar Pro":
#     perplexity_api_key = st.sidebar.text_input("Perplexity API Key", type="password")
#     # Store API keys in session state
if groq_api_key and firecrawl_api_key and tavily_api_key:
    st.session_state.groq_api_key = groq_api_key
    st.session_state.firecrawl_api_key = firecrawl_api_key
    st.session_state.tavily_api_key = tavily_api_key

# else:
#     st.sidebar.warning("Please enter all required API keys to proceed.")
# else:  # Exa AI
#     exa_api_key = st.sidebar.text_input("Exa API Key", type="password")
#     # Store API keys in session state
#     if openai_api_key and firecrawl_api_key and brave_api_key:
#         st.session_state.openai_api_key = openai_api_key
#         st.session_state.firecrawl_api_key = firecrawl_api_key
#         st.session_state.brave_api_key = brave_api_key
#     else:
#         st.sidebar.warning("Please enter all required API keys to proceed.")

# Main UI
st.title("Company Analysis Agent")
st.info(
    """
    This agent helps analyze competitors by extracting structured data from competitor websites and generating insights using AI.
    - Provide a **URL** or a **description** of target company.
    - The app will fetch competitor URLs, extract relevant information, and generate a detailed analysis report.
    - For better results, provide both URL and a 5-6 word description of the company.
    """
)

# Input fields for URL and description
url = st.text_input("Enter target company URL :")
description = st.text_area("Enter a description of the company (if URL is not available):")

# Initialize API keys and tools
if (
        "groq_api_key" in st.session_state and
        "firecrawl_api_key" in st.session_state and
        "tavily_api_key" in st.session_state
):
    # Initialize Firecrawl tool for crawling competitor websites
    firecrawl_tools = FirecrawlTools(
        api_key=st.session_state.firecrawl_api_key,
        scrape=False,
        crawl=True,
        limit=5
    )

    # Agent to fetch and extract competitor website content
    firecrawl_agent = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Agent to generate the competitor analysis report
    analysis_agent = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Agent to compare competitor data (e.g., create structured comparison table)
    comparison_agent = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


    def get_competitor_urls(url: str = None, description: str = None) -> list[str]:
        if not url and not description:
            raise ValueError("Please provide either a URL or a description.")

        # query = "Find websites that provide detailed market and competitive analysis of products or services"
        #
        # if url and description:
        #     query += f" for a company with website +{url} and described as: +{description}."
        # elif url:
        #     query += f" for the company with website +{url}."
        # else:
        #     query += f" for a company described as: +{description}."
        #
        # # Prioritizing high-quality sources
        # query += (
        #     " Only return URLs from trusted sources such as Gartner, Forrester, Bloomberg, McKinsey, IDC, or official company pages."
        #     " Exclude general blogs, forums, or unrelated reports."
        # )
        query = "List websites that offer more detailed information on analysis of products or services"
        if url and description:
            query += f" to a company with website {url} and description: {description}."
        elif url:
            query += f" to the company with website {url}."
        else:
            query += f" to a company described as: {description}."
        query += " Only return the URL with most relevant information."

        # Tavily API Setup
        tavily_wrapper = TavilySearchAPIWrapper(tavily_api_key=st.session_state.tavily_api_key)

        try:
            web_links = tavily_wrapper.results(query, max_results=5)  # Fetch results

            # Extract URLs and filter duplicates or irrelevant results
            filtered_urls = list(set(result['url'] for result in web_links if 'url' in result))
            filtered_urls = [url for url in filtered_urls if not any(excl in url for excl in ['blog', 'forum', 'news'])]

            return filtered_urls

        except Exception as e:
            print(f"Error fetching competitor URLs: {e}")
            return []


    class CompetitorDataSchema(BaseModel):
        company_name: str = Field(description="Name of the company, Launch Date, Sequence of Features addition")
        # When did it start, Revenue(Company, Product), Market Share
        pricing: str = Field(description="Pricing details, tiers, and plans")  # Price trend
        key_features: List[str] = Field(description="Main features and capabilities of the product/service")
        tech_stack: List[str] = Field(
            description="Technologies, frameworks, and tools used")  # Innovation Focus stratergy
        marketing_focus: str = Field(description="Main marketing angles and target audience,Segement, Industry")
        # Segement, Industry, Company Size(No of employee), Revenue,
        customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")
        # Name of Customer of competitor, Competitors of customers of our competitor(probable prospects), Ease of Customer Engagement
        # Strength and weakness- tool


    def extract_competitor_info(competitor_url: str) -> Optional[dict]:
        try:
            # Initialize FirecrawlApp with API key
            app = FirecrawlApp(api_key=st.session_state.firecrawl_api_key)

            # Add wildcard to crawl subpages
            url_pattern = f"{competitor_url}/*"

            extraction_prompt = """
                Extract detailed information about the company's offerings, including:
                - Company name and basic information,Launch Date, Sequence of Features addition,Revenue, Market Share
                - Pricing details, plans, and tiers
                - Key features and main capabilities
                - Technology stack and technical details and Innovation Focus strategy
                - Marketing focus , target audience, Market Segment, Industry and Company Size
                - Customer feedback and testimonials

                Analyze the entire website content to provide comprehensive information for each field.
                """

            response = app.extract(
                [url_pattern],
                {
                    'prompt': extraction_prompt,
                    'schema': CompetitorDataSchema.model_json_schema(),
                }
            )

            if response.get('success') and response.get('data'):
                extracted_info = response['data']

                # Create JSON structure
                competitor_json = {
                    "competitor_url": competitor_url,
                    "company_name": extracted_info.get('company_name', 'N/A'),
                    "pricing": extracted_info.get('pricing', 'N/A'),
                    "key_features": extracted_info.get('key_features', [])[:5],  # Top 5 features
                    "tech_stack": extracted_info.get('tech_stack', [])[:5],  # Top 5 tech stack items
                    "marketing_focus": extracted_info.get('marketing_focus', 'N/A'),
                    "customer_feedback": extracted_info.get('customer_feedback', 'N/A')
                }

                return competitor_json

            else:
                return None

        except Exception as e:
            return None


    def generate_comparison_report(competitor_data: list) -> None:
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        print(formatted_data)

        # Updated system prompt for more structured output
        system_prompt = f"""
            As an expert business analyst, analyze the following data in JSON format and create a structured report.
            Extract and summarize the key information into concise points. For multiple data sources chose the most relevant one and fill the table

            {formatted_data}

            Return the data in JSON format with exactly these keys:
            [
                {{
                    "Company": "Company Name (URL)",
                    "Pricing": "Pricing details",
                    "Key Features": "Feature 1", "Feature 2", "Feature 3"
                    "Tech Stack": "Tech 1", "Tech 2", "Tech 3"
                    "Marketing Focus": "Marketing focus summary",
                    "Customer Feedback": "Brief customer feedback summary"
                }},
                ...
            ]

            Ensure all fields contain meaningful data. If information is missing, return "N/A" instead of leaving it empty. All the websites refer to the same Product, so just one row in the table.




            """

        # Get comparison data from agent
        comparison_response = comparison_agent.invoke(system_prompt)

        try:
            # Get comparison data from the agent
            comparison_response = comparison_agent.invoke(system_prompt)

            # Convert model output to JSON format
            comparison_data = json.loads(comparison_response.content)

            # Validate JSON structure
            if not isinstance(comparison_data, list) or not all(isinstance(entry, dict) for entry in comparison_data):
                raise ValueError("Invalid JSON format received from model.")

            # Convert JSON to DataFrame
            df = pd.DataFrame(comparison_data)

            # Display the table
            st.subheader("Competitor Comparison")
            st.table(df)

        except Exception as e:
            st.error(f"Error creating comparison table: {str(e)}")
            st.write("Raw comparison data for debugging:", comparison_response.content)

        # try:
        #     # Split the response into lines and clean them
        #     table_lines = [
        #         line.strip()
        #         for line in comparison_response.content.split('\n')
        #         if line.strip() and '|' in line
        #     ]
        #
        #     # Extract headers (first row)
        #     headers = [
        #         col.strip()
        #         for col in table_lines[0].split('|')
        #         if col.strip()
        #     ]
        #
        #     # Extract data rows (skip header and separator rows)
        #     data_rows = []
        #     for line in table_lines[2:]:  # Skip header and separator rows
        #         row_data = [
        #             cell.strip()
        #             for cell in line.split('|')
        #             if cell.strip()
        #         ]
        #         if len(row_data) == len(headers):
        #             data_rows.append(row_data)
        #
        #     # Create DataFrame
        #     df = pd.DataFrame(
        #         data_rows,
        #         columns=headers
        #     )
        #
        #     # Display the table
        #     st.subheader("Competitor Comparison")
        #     st.table(df)

        # except Exception as e:
        #     st.error(f"Error creating comparison table: {str(e)}")
        #     st.write("Raw comparison data for debugging:", comparison_response.content)


    def generate_analysis_report(competitor_data: list):
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        print("Analysis Data:", formatted_data)  # For debugging

        report = analysis_agent.invoke(
            f"""Analyze the following data in JSON format and identify market opportunities to improve:
            # Add data from GPT

                {formatted_data}

                Tasks:
                Competitor Landscape Analysis
                Identify the market and analyze their product offerings, pricing strategies, and market positioning. Collect data on their strengths, weaknesses, and customer feedback to understand their impact on the industry.

                Unmet Customer Needs & Market Gaps
                Analyze customer pain points and unmet needs by examining industry trends, reviews, and feedback. Identify areas where existing products or services fall short and where demand is not being fully met.

                Consumer Behavior & Preferences
                Gather data on customer demographics, buying behaviors, and key decision-making factors. Analyze trends in consumer expectations and preferences that influence purchase decisions in this market.

                Pricing & Positioning Insights
                Compare competitor pricing strategies and assess their value propositions. Identify trends in premium, mid-range, and budget offerings, and determine how different price points affect customer perception.

                Emerging Industry Trends & Technological Shifts
                Research upcoming trends, innovations, and technological advancements that the company plans. Analyze how these changes impact market dynamics and what future opportunities they might create.

                Growth Opportunities in Underserved Segments
                Identify niche or underserved customer segments with potential for growth. Analyze barriers to entry, customer demand, and competitive gaps that could create new opportunities.

                Key Takeaways with Strategic Recommendations
                Summarize the major findings from the research and highlight key insights. Provide a few high-level suggestions based on data, focusing on areas where differentiation could be achieved.
                """
        )
        return report.content


    # Run analysis when the user clicks the button
    if st.button("Analyze Product"):
        if url or description:
            with st.spinner("Fetching competitor URLs..."):
                competitor_urls = get_competitor_urls(url=url, description=description)
                st.write(f"Competitor URLs: {competitor_urls}")

            competitor_data = []
            for comp_url in competitor_urls:
                with st.spinner(f"Analyzing Competitor: {comp_url}..."):
                    competitor_info = extract_competitor_info(comp_url)
                    if competitor_info is not None:
                        competitor_data.append(competitor_info)

            if competitor_data:
                # Generate and display comparison report
                with st.spinner("Generating comparison table..."):
                    generate_comparison_report(competitor_data)

                # Generate and display final analysis report
                with st.spinner("Generating analysis report..."):
                    analysis_report = generate_analysis_report(competitor_data)
                    st.subheader("Competitor Analysis Report")
                    st.markdown(analysis_report)

                st.success("Analysis complete!")
            else:
                st.error("Could not extract data from any competitor URLs")
        else:
            st.error("Please provide either a URL or a description.")