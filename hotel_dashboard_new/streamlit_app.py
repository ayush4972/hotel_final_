import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Try to import existing processing from pages if available
try:
    from pages import sentiment_analysis as sa
except Exception:
    sa = None


def show_home():
    st.title("🏨 Hotel Reservation Analytics — Streamlit")
    st.markdown(
        """
        This lightweight Streamlit wrapper exposes a few views from the existing Dash project so
        you can deploy quickly on Streamlit Cloud or run locally.

        Use the sidebar to navigate. The Sentiment Analysis page reuses the project code located
        in `pages/sentiment_analysis.py` (if present) and shows interactive Plotly charts.
        """
    )

    st.header("Quick actions")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Dataset (raw)", "Hotel Reservations.csv")
    with col2:
        modified = Path("Hotel_Reservations_Modified.csv")
        st.metric("Modified dataset", "Present" if modified.exists() else "Not created")

    st.info("Run `streamlit run streamlit_app.py` to start this app locally.")


def show_sentiment():
    st.title("💝 Sentiment Analysis")

    if sa is None:
        st.error("The module `pages.sentiment_analysis` could not be imported. Make sure you're running this from the project root and the `pages` package exists.")
        return

    # Access the processed dataframe if available
    df = getattr(sa, "df_limited", None)

    if df is None or df.empty:
        st.warning("No sentiment data is available. The Dash module may have failed to create the processed dataset. Check console logs.")
    else:
        st.subheader("Dataset sample")
        st.dataframe(df.head(20))

        st.subheader("Sentiment Distribution")
        try:
            fig = sa.create_sentiment_distribution()
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render sentiment distribution: {e}")

        st.subheader("Sentiment vs Booking Status")
        try:
            fig2 = sa.create_sentiment_vs_status()
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render sentiment vs status: {e}")

        st.subheader("Average Price by Sentiment")
        try:
            fig3 = sa.create_sentiment_vs_price()
            st.plotly_chart(fig3, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render sentiment vs price: {e}")

        st.subheader("Sentiment Trend Over Time")
        try:
            fig4 = sa.create_sentiment_trend()
            st.plotly_chart(fig4, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to render sentiment trend: {e}")

        st.markdown("---")
        st.write("You can download the processed CSV (if available):")
        modified_path = Path("Hotel_Reservations_Modified.csv")
        if modified_path.exists():
            with open(modified_path, "rb") as fh:
                st.download_button("Download modified CSV", fh, file_name="Hotel_Reservations_Modified.csv")
        else:
            st.info("`Hotel_Reservations_Modified.csv` not found. The Dash preprocessing step typically creates this file on first run.")


def main():
    st.set_page_config(page_title="Hotel Analytics (Streamlit)", layout="wide")

    pages = {
        "Home": show_home,
        "Sentiment Analysis": show_sentiment,
    }

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(pages.keys()))

    # Run the page
    pages[choice]()


if __name__ == "__main__":
    main()
