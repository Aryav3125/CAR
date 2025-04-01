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

if "groq_api_key" in st.session_state:
    os.environ["GROQ_API_KEY"] = st.session_state.groq_api_key
# Streamlit UI
st.set_page_config(page_title="Competitor Analysis Agent", layout="wide")

# Sidebar for API keys
st.sidebar.title("API Keys")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
firecrawl_api_key = st.sidebar.text_input("Firecrawl API Key", type="password")
brave_api_key = st.sidebar.text_input("Brave Search API Key", type="password")

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
if groq_api_key and firecrawl_api_key and brave_api_key:
    st.session_state.groq_api_key = groq_api_key
    st.session_state.firecrawl_api_key = firecrawl_api_key
    st.session_state.brave_api_key = brave_api_key


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
st.title("Competitor Analysis Agent")
st.info(
    """
    This agent helps analyze competitors by extracting structured data from competitor websites and generating insights using AI.
    - Provide a **URL** or a **description** of your company.
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
        "brave_api_key" in st.session_state
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
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Agent to generate the competitor analysis report
    analysis_agent = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # Agent to compare competitor data (e.g., create structured comparison table)
    comparison_agent = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


    def get_competitor_urls(url: str = None, description: str = None) -> list[str]:
        if not url and not description:
            raise ValueError("Please provide either a URL or a description.")

        # Extract domain from the URL (if provided) to exclude it from results
        domain = url.split("//")[-1].split("/")[0] if url else None

        # Improved Query for more relevant competitors
        query = "List top direct competitor company website that offer similar products or services"

        if url and description:
            query += f" to a company with website {url} and description: {description}."
        elif url:
            query += f" to the company with website {url}."
        else:
            query += f" to a company described as: {description}."

        query += " Exclude results from the same domain. Only return the main homepages of competitors, not subpages or directories."

        # Set up the Brave Search API call
        api_url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": st.session_state.brave_api_key
        }
        # Pass the query in the 'q' parameter; adjust 'count' if supported.
        params = {
            "q": query,
            "count": 1
        }
        # print(headers)
        # print("Request headers:", headers)
        # response = requests.get(api_url, headers=headers, params=params)

        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()

            # Parse competitor URLs from the response.
            # Assuming the results are under data["web"]["results"] and each result contains a "url" key.
            urls = []
            for result in data.get("web", {}).get("results", []):
                url_result = result.get("url") or result.get("link")
                if url_result and (not domain or domain not in url_result):
                    urls.append(url_result)
            return urls[:1]
        except Exception as e:
            st.error(f"Error fetching competitor URLs from Brave Search API: {str(e)}")
            return []


    class CompetitorDataSchema(BaseModel):
        company_name: str = Field(description="Name of the company, Launch Date, Sequence of Features addition")
        #When did it start, Revenue(Company, Product), Market Share
        pricing: str = Field(description="Pricing details, tiers, and plans") #Price trend
        key_features: List[str] = Field(description="Main features and capabilities of the product/service")
        tech_stack: List[str] = Field(description="Technologies, frameworks, and tools used") #Innovation Focus stratergy
        marketing_focus: str = Field(description="Main marketing angles and target audience,Segement, Industry")
        #Segement, Industry, Company Size(No of employee), Revenue,
        customer_feedback: str = Field(description="Customer testimonials, reviews, and feedback")
        #Name of Customer of competitor, Competitors of customers of our competitor(probable prospects), Ease of Customer Engagement
        #Strength and weakness- tool



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
            As an expert business analyst, analyze the following competitor data in JSON format and create a structured comparison.
            Extract and summarize the key information into concise points.

            {formatted_data}

            Return the data in a structured format with EXACTLY these columns:
            Company, Pricing, Key Features, Tech Stack, Marketing Focus, Customer Feedback

            Rules:
            1. For Company: Include company name and URL
            2. For Key Features: List top 3 most important features only
            3. For Tech Stack: List top 3 most relevant technologies only
            4. Keep all entries clear and concise
            5. Format feedback as brief quotes
            6. Return ONLY the structured data, no additional text
            """

        # Get comparison data from agent
        comparison_response = comparison_agent.invoke(system_prompt)

        try:
            # Split the response into lines and clean them
            table_lines = [
                line.strip()
                for line in comparison_response.content.split('\n')
                if line.strip() and '|' in line
            ]

            # Extract headers (first row)
            headers = [
                col.strip()
                for col in table_lines[0].split('|')
                if col.strip()
            ]

            # Extract data rows (skip header and separator rows)
            data_rows = []
            for line in table_lines[2:]:  # Skip header and separator rows
                row_data = [
                    cell.strip()
                    for cell in line.split('|')
                    if cell.strip()
                ]
                if len(row_data) == len(headers):
                    data_rows.append(row_data)

            # Create DataFrame
            df = pd.DataFrame(
                data_rows,
                columns=headers
            )

            # Display the table
            st.subheader("Competitor Comparison")
            st.table(df)

        except Exception as e:
            st.error(f"Error creating comparison table: {str(e)}")
            st.write("Raw comparison data for debugging:", comparison_response.content)


    def generate_analysis_report(competitor_data: list):
        # Format the competitor data for the prompt
        formatted_data = json.dumps(competitor_data, indent=2)
        print("Analysis Data:", formatted_data)  # For debugging

        report = analysis_agent.invoke(
            f"""Analyze the following competitor data in JSON format and identify market opportunities to improve my own company:
                
                {formatted_data}

                Tasks:
                Competitor Landscape Analysis
                Identify key competitors in the market and analyze their product offerings, pricing strategies, and market positioning. Collect data on their strengths, weaknesses, and customer feedback to understand their impact on the industry.

                Unmet Customer Needs & Market Gaps
                Analyze customer pain points and unmet needs by examining industry trends, reviews, and feedback. Identify areas where existing products or services fall short and where demand is not being fully met.

                Consumer Behavior & Preferences
                Gather data on customer demographics, buying behaviors, and key decision-making factors. Analyze trends in consumer expectations and preferences that influence purchase decisions in this market.

                Pricing & Positioning Insights
                Compare competitor pricing strategies and assess their value propositions. Identify trends in premium, mid-range, and budget offerings, and determine how different price points affect customer perception.

                Emerging Industry Trends & Technological Shifts
                Research upcoming trends, innovations, and technological advancements shaping the industry. Analyze how these changes impact market dynamics and what future opportunities they might create.

                Growth Opportunities in Underserved Segments
                Identify niche or underserved customer segments with potential for growth. Analyze barriers to entry, customer demand, and competitive gaps that could create new opportunities.

                Key Takeaways with Minimal Strategic Recommendations
                Summarize the major findings from the research and highlight key insights. Provide a few high-level suggestions based on data, focusing on areas where differentiation could be achieved.
                """
        )
        return report.content


    # Run analysis when the user clicks the button
    if st.button("Analyze Competitors"):
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
