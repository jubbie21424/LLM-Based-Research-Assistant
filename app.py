# Streamlit framework for building web applications
import streamlit as st

# Wikipedia retriever from LangChain for fetching Wikipedia articles
from langchain_community.retrievers import WikipediaRetriever

# Google Gemini LLM integration from LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Message types for structuring prompts to the LLM
from langchain_core.messages import SystemMessage, HumanMessage

# Regular expressions library used for input validation and citation handling (word count excludes bracketed citations)
import re

# =============================================================================
# Q0: Setup API Key Configuration
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    st.selectbox("Select LLM", ["Gemini 2.5 Flash"])
    api_key = st.text_input("Enter Gemini API Key", type="password")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def clean_text(text: str) -> str:
    """Light cleaning to prevent encoding/display issues."""
    if not text:
        return ""
    replacements = {
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-",
        "\u2212": "-", "\u2026": "...",
        "\u00a0": " ", "\u00ad": ""
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return " ".join(text.split())


def build_sources_text(docs, excerpt_len=1200):
    """Prepare Wikipedia source text (truncate to avoid token overflow)."""
    sources = []
    for d in docs:
        title = clean_text(d.metadata.get("title", ""))
        url = d.metadata.get("source", "")
        excerpt = clean_text((d.page_content or "")[:excerpt_len])
        sources.append(f"Title: {title}\nURL: {url}\nExcerpt: {excerpt}")
    return "\n\n".join(sources)


def is_gibberish_industry(industry: str) -> bool:
    """
    Simple gibberish filter for Q1.

    Treat as gibberish if:
    - empty after strip, OR
    - does NOT contain at least one English letter OR one CJK character.
    This catches inputs like: "!!!@@@", "12345", "....", "%%^&*"
    and still allows: "finance", "AI", "餐飲", "電商", "FinTech".
    """
    s = (industry or "").strip()
    if s == "":
        return True
    return re.search(r"[A-Za-z\u4e00-\u9fff]", s) is None

def validate_is_industry_llm(user_input: str, api_key: str) -> bool:
    """
    Strictly validates if the input is a business industry, sector, or market.
    Rejects names of people, specific products (food/gadgets), 
    locations, or non-commercial topics.
    """
    try:
        checker_llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.0
        )
        
        # Refined prompt for business context
        system_instruction = (
        "You are a strict Business Classification Guard. Your ONLY job is to filter out inputs that are NOT general business industries.\n\n"
            "STRICT RULES:\n"
            "1. REJECT (Answer 'No'):\n"
            "   - Specific people (e.g., Elon Musk, Jay Chou, Taylor Swift)\n"
            "   - Specific products or brands (e.g., iPhone, Big Mac, Coca-Cola)\n"
            "   - Specific locations/addresses (e.g., London, No. 10 Downing St, Taiwan)\n"
            "   - Movies, songs, or fictional characters.\n"
            "   - Anything that is a single object (e.g., Apple, Pizza, Car).\n"
            "2. ACCEPT (Answer 'Yes'):\n"
            "   - Broad economic sectors (e.g., Automotive Industry, Semiconductor, Fast Food Sector, Healthcare).\n\n"
            "If the input is borderline, favor 'No'. Answer ONLY 'Yes' or 'No'."
        )
        
        user_msg = f"Term: {user_input}\nIs this a business industry? Answer 'Yes' or 'No' only."
        
        # Combine for the call
        response = checker_llm.invoke([
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_msg)
        ])
        
        return "yes" in response.content.strip().lower()
    except Exception:
        # Fallback to True to ensure app doesn't crash
        return True
# =============================================================================
# Q1: TEXT & INPUTS
# =============================================================================
st.title("MSIN0231-Individual assignment")
st.header("Market Research Assistant")
st.markdown("**Candidate No:** VTVS0")

# Check API key
if not api_key:
    st.warning("👈 Please enter your Gemini API Key in the sidebar to continue")
    st.stop()

# init state
if "industry_input" not in st.session_state:
    st.session_state["industry_input"] = ""


def clear_industry():
    st.session_state["industry_input"] = ""


# Input (controlled by session_state)
industry = st.text_input("Please provide an industry", key="industry_input")

# Buttons row (Search + Clear)
col1, col2, _ = st.columns([1, 1, 6])

with col1:
    search_clicked = st.button("Search", type="primary", use_container_width=True)

with col2:
    st.button("Clear", on_click=clear_industry, type="secondary", use_container_width=True)

# Run when Search clicked
if search_clicked:
    industry_clean = (industry or "").strip()

    # Step 1: Basic empty check
    if industry_clean == "":
        st.warning("⚠️ The industry has not been provided. Please update.")
        st.stop()

    # Step 2: Basic regex gibberish check (Fast local validation to prevent unnecessary API calls and token usage)
    if is_gibberish_industry(industry_clean):
        st.warning("⚠️ Invalid industry input (contains no letters or CJK characters).")
        st.stop()

    # Step 3: Semantic validation via LLM (intercepting nonsense like 'jfepwfjwpwv')
    with st.spinner("Validating business context..."):
        if not validate_is_industry_llm(industry_clean, api_key):
            st.error(f"❌ '{industry_clean}' is not a recognized industry. Please enter a sector like 'Fintech' or 'Tourism'.")
            st.stop()

    st.success(f"Verified Industry: **{industry_clean}**")
    # =============================================================================
    # Q2: Return URL
    # =============================================================================
    st.subheader("Top 5 Relevant Wikipedia Pages:")

    with st.spinner("Searching Wikipedia..."):
        try:
            retriever = WikipediaRetriever(top_k_results=5)
            docs = retriever.invoke(industry_clean)

            if not docs:
                st.error("No Wikipedia pages found for this industry. Please try a different term.")
                st.stop()

            if len(docs) < 5:
                st.warning(f"Note: Only {len(docs)} Wikipedia page(s) found for this industry (requested 5).")

            for i, doc in enumerate(docs, 1):
                st.markdown(f"{i}. [{doc.metadata.get('title','Untitled')}]({doc.metadata.get('source','')})")

        except Exception as e:
            st.error(f"Error retrieving Wikipedia data: {str(e)}")
            st.info("Please try again in a moment or try a different industry term.")
            st.stop()

    # =============================================================================
    # Q3: Return report
    # =============================================================================
    sources_text = build_sources_text(docs)
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.1
        )

        # System prompt
        system_content = clean_text(
            "You are a professional market research analyst producing a structured business report. "
            "Base your analysis strictly and exclusively on the provided Wikipedia excerpts. "
            "Do not introduce external knowledge or assumptions. "
            "If a detail is not stated in the sources, write: Not stated in the sources. "
            "Write in a professional, analytical, and objective tone. "
            "CRITICAL: The main content must be MAXIMUM 450 words (excluding citations in brackets). "
            "Keep each section brief and focused on key insights only. Prioritize conciseness."
        )

        system_prompt = SystemMessage(content=system_content)

        # User prompt
        user_content = clean_text(
            f"Industry: {industry_clean}\n\n"
            "Write a CONCISE industry report (MAX 450 words excluding citations) with these sections:\n"
            "1) Industry Overview (80 words max)\n"
            "2) Value Chain Analysis (80 words max)\n"
            "3) Key Trends and Opportunities (100 words max)\n"
            "4) Challenges and Risks (100 words max)\n"
            "5) Strategic Outlook (90 words max)\n\n"
            "Use bullet points where appropriate to save space.\n"
            "Cite sources using brackets [Page Title] at the end of each section.\n\n"
            f"Sources (Wikipedia excerpts):\n{sources_text}"
        )

        user_prompt = HumanMessage(content=user_content)

        st.subheader("Industry Report:")

        # Multiple attempts to get under 500 words (brief requirement)
        max_attempts = 3
        attempt = 0
        word_count = 0

        while attempt < max_attempts:
            attempt += 1

            with st.spinner(f"Generating report (Attempt {attempt}/{max_attempts})..."):
                response = llm.invoke([system_prompt, user_prompt])

            # Count words excluding bracketed citations, e.g., [Page Title]
            content_wo_citations = re.sub(r"\[[^\]]*\]", "", response.content)
            word_count = len(content_wo_citations.split())

            if word_count <= 500:
                break
            else:
                if attempt < max_attempts:
                    st.warning(f"⚠️ Attempt {attempt}: {word_count} words. Regenerating...")

        st.caption(f"Word count: {word_count}/500 (excluding citations)")

        st.write(response.content)

    except Exception as e:
        # Error messages:
        if "api key" in str(e).lower():
            st.info("⚠️ Please check your API key is valid.")
        elif "quota" in str(e).lower():
            st.info("⚠️ API quota exceeded. Please try again later.")
        else:
            st.info("⚠️ Please check your API key and try again.")
