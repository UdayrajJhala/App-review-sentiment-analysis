import gradio as gr
import torch
from transformers import AutoTokenizer, BertForSequenceClassification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
from google_play_scraper import app, reviews, Sort
from datetime import datetime
import warnings
import base64
from io import BytesIO
import re
import urllib.parse
import tempfile

warnings.filterwarnings("ignore")

# Load the trained model and tokenizer
print("Loading model...")
model_path = "./bert_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
print(f"Model loaded successfully on {device}!")

# Sentiment mappings
SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}
SENTIMENT_EMOJI = {0: "😞", 1: "😐", 2: "😊"}
SENTIMENT_COLORS = {0: "#FF6B6B", 1: "#FFD93D", 2: "#6BCF7F"}  # Red, Yellow, Green


def encode(text):
    """Tokenize a single text (replaces the removed tokenizer.encode_plus)"""
    return tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )


def predict_sentiment(review_text):
    """Predict sentiment for a single review"""
    if not review_text or review_text.strip() == "":
        return "Please enter a review to analyze.", ""

    encoding = encode(review_text)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    # Create confidence scores text
    confidence_text = ""
    for idx, label in SENTIMENT_LABELS.items():
        emoji = SENTIMENT_EMOJI[idx]
        prob = float(probabilities[0][idx].item())
        confidence_text += f"{emoji} {label}: {prob*100:.2f}%\n"

    sentiment = SENTIMENT_LABELS[predicted_class]
    emoji = SENTIMENT_EMOJI[predicted_class]

    result_html = f"""
    <div style="text-align: center; padding: 20px; border-radius: 10px; background: linear-gradient(135deg, {SENTIMENT_COLORS[predicted_class]}33, {SENTIMENT_COLORS[predicted_class]}66);">
        <h1 style="font-size: 4em; margin: 10px;">{emoji}</h1>
        <h2 style="color: {SENTIMENT_COLORS[predicted_class]}; margin: 10px; font-weight: bold;">{sentiment}</h2>
        <p style="font-size: 1.2em; color: #FFFFFF;">Confidence: {confidence*100:.2f}%</p>
    </div>
    """

    return result_html, confidence_text


def predict_batch(reviews_list):
    """Predict sentiment for multiple reviews"""
    predictions = []

    for review_text in reviews_list:
        if not review_text or review_text.strip() == "":
            predictions.append(1)  # neutral for empty
            continue

        encoding = encode(review_text)

        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class)

    return predictions


def extract_package_name(url_or_package):
    """Extract package name from Google Play Store URL or return as-is if already a package name"""
    if not url_or_package or url_or_package.strip() == "":
        return ""

    url_or_package = url_or_package.strip()

    # If it looks like a URL, extract the package name
    if "play.google.com" in url_or_package or url_or_package.startswith("http"):
        try:
            if "id=" in url_or_package:
                # Extract everything after id=
                package_name = url_or_package.split("id=")[1]
                # Remove any additional parameters
                if "&" in package_name:
                    package_name = package_name.split("&")[0]
                return package_name
        except Exception:
            pass

    # If it doesn't contain dots, it's probably not a valid package name
    if "." not in url_or_package:
        return ""

    # Otherwise, assume it's already a package name
    return url_or_package


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100, facecolor="white")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return f"data:image/png;base64,{img_str}"


def scrape_and_analyze(app_url_or_id, num_reviews=100):
    """Scrape reviews from Google Play Store and analyze them"""
    try:
        # Extract package name from URL or use as-is
        app_id = extract_package_name(app_url_or_id)

        # Validate inputs
        if not app_id or app_id.strip() == "":
            return "Please enter a valid Google Play Store URL or app package name.", "", ""

        num_reviews = int(num_reviews)
        if num_reviews < 10 or num_reviews > 500:
            return "Number of reviews must be between 10 and 500.", "", ""

        # Scrape app info
        status_msg = f"📱 Fetching app information for: {app_id}..."
        print(status_msg)

        try:
            app_info = app(app_id, lang="en", country="us")
            app_name = app_info.get("title", "Unknown App")
            app_rating = app_info.get("score", "N/A")
        except Exception as e:
            return (
                f"❌ Error: Could not find app with ID '{app_id}'. Please check the app ID and try again.\n\nTip: App ID is the part after 'id=' in the Play Store URL.",
                "",
                "",
            )

        status_msg = f"📥 Scraping {num_reviews} reviews for {app_name}..."
        print(status_msg)

        # Scrape reviews
        result, _ = reviews(
            app_id, lang="en", country="us", sort=Sort.NEWEST, count=num_reviews
        )

        if not result:
            return "No reviews found for this app.", "", ""

        # Extract review texts
        review_texts = [r["content"] for r in result if r["content"]]
        review_scores = [r["score"] for r in result if r["content"]]
        review_dates = [r["at"] for r in result if r["content"]]

        status_msg = f"🤖 Analyzing {len(review_texts)} reviews with BERT..."
        print(status_msg)

        # Predict sentiments with confidence scores
        predictions = []
        confidence_scores = []

        for review_text in review_texts:
            if not review_text or review_text.strip() == "":
                predictions.append(1)  # neutral for empty
                confidence_scores.append(0.5)
                continue

            encoding = encode(review_text)

            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

                predictions.append(predicted_class)
                confidence_scores.append(confidence)

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Review": review_texts,
                "Star_Rating": review_scores,
                "Date": review_dates,
                "AI_Sentiment": [SENTIMENT_LABELS[p] for p in predictions],
                "Confidence_Score": [f"{c*100:.1f}%" for c in confidence_scores],
                "Sentiment_Code": predictions,
            }
        )

        # Calculate statistics
        sentiment_counts = df["AI_Sentiment"].value_counts()
        total = len(df)

        # Create visualizations
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        labels = [
            f"{SENTIMENT_EMOJI[i]} {label}" for i, label in SENTIMENT_LABELS.items()
        ]
        values = [sentiment_counts.get(label, 0) for label in SENTIMENT_LABELS.values()]
        colors = [SENTIMENT_COLORS[i] for i in range(3)]

        wedges, texts, autotexts = ax1.pie(
            values,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"size": 11, "weight": "bold"},
        )
        ax1.set_title("Sentiment Distribution", fontsize=14, weight="bold", pad=20)

        # Bar chart by star rating
        sentiment_by_stars = (
            df.groupby(["Star_Rating", "AI_Sentiment"]).size().unstack(fill_value=0)
        )

        # Create color mapping based on actual column order
        column_colors = []
        for col in sentiment_by_stars.columns:
            if col == "Negative":
                column_colors.append(SENTIMENT_COLORS[0])  # Red
            elif col == "Neutral":
                column_colors.append(SENTIMENT_COLORS[1])  # Yellow
            elif col == "Positive":
                column_colors.append(SENTIMENT_COLORS[2])  # Green

        sentiment_by_stars.plot(
            kind="bar",
            ax=ax2,
            color=column_colors,
        )
        ax2.set_title("Sentiment by Star Rating", fontsize=14, weight="bold", pad=20)
        ax2.set_xlabel("Star Rating", fontsize=11)
        ax2.set_ylabel("Number of Reviews", fontsize=11)
        ax2.legend(title="Sentiment", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=0)

        plt.tight_layout()

        # Convert to base64
        chart_img = fig_to_base64(fig)

        # Create summary HTML with embedded chart
        summary_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 25px; border-radius: 15px; color: white; margin-bottom: 20px;">
            <h2 style="margin: 0 0 15px 0;">📊 Analysis Summary: {app_name}</h2>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-top: 15px;">
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                    <h3 style="margin: 0; font-size: 1.1em;">⭐ Play Store Rating</h3>
                    <p style="font-size: 2em; margin: 10px 0 0 0; font-weight: bold;">{app_rating}/5</p>
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px;">
                    <h3 style="margin: 0; font-size: 1.1em;">📝 Reviews Analyzed</h3>
                    <p style="font-size: 2em; margin: 10px 0 0 0; font-weight: bold;">{total}</p>
                </div>
            </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-top: 20px;">
            <div style="background: {SENTIMENT_COLORS[2]}33; padding: 20px; border-radius: 10px; border-left: 5px solid {SENTIMENT_COLORS[2]};">
                <h3 style="margin: 0; color: {SENTIMENT_COLORS[2]};">😊 Positive</h3>
                <p style="font-size: 2em; margin: 10px 0 5px 0; font-weight: bold;">{sentiment_counts.get('Positive', 0)}</p>
                <p style="margin: 0; color: #666;">{sentiment_counts.get('Positive', 0)/total*100:.1f}%</p>
            </div>
            <div style="background: {SENTIMENT_COLORS[1]}33; padding: 20px; border-radius: 10px; border-left: 5px solid {SENTIMENT_COLORS[1]};">
                <h3 style="margin: 0; color: #F4A404;">😐 Neutral</h3>
                <p style="font-size: 2em; margin: 10px 0 5px 0; font-weight: bold;">{sentiment_counts.get('Neutral', 0)}</p>
                <p style="margin: 0; color: #666;">{sentiment_counts.get('Neutral', 0)/total*100:.1f}%</p>
            </div>
            <div style="background: {SENTIMENT_COLORS[0]}33; padding: 20px; border-radius: 10px; border-left: 5px solid {SENTIMENT_COLORS[0]};">
                <h3 style="margin: 0; color: {SENTIMENT_COLORS[0]};">😞 Negative</h3>
                <p style="font-size: 2em; margin: 10px 0 5px 0; font-weight: bold;">{sentiment_counts.get('Negative', 0)}</p>
                <p style="margin: 0; color: #666;">{sentiment_counts.get('Negative', 0)/total*100:.1f}%</p>
            </div>
        </div>
        
        <div style="margin-top: 30px; background: white; padding: 20px; border-radius: 10px;">
            <img src="{chart_img}" style="width: 100%; border-radius: 10px;"/>
        </div>
        """

        # Create sample reviews text (first 20 reviews)
        sample_reviews = f"📝 Sample Reviews (Showing 20 out of {total} reviews):\n"
        sample_reviews += f"💾 Download the CSV file below to get all {total} reviews with their sentiments!\n\n"
        sample_reviews += "=" * 80 + "\n\n"

        sample_df = df.head(20)
        for idx, (index, row) in enumerate(sample_df.iterrows(), 1):
            emoji = SENTIMENT_EMOJI[
                [k for k, v in SENTIMENT_LABELS.items() if v == row["AI_Sentiment"]][0]
            ]
            date_str = (
                row["Date"].strftime("%Y-%m-%d")
                if hasattr(row["Date"], "strftime")
                else str(row["Date"])
            )

            sample_reviews += f"Review #{idx} | {date_str} | ⭐ {row['Star_Rating']}/5 | {emoji} {row['AI_Sentiment']}\n"
            sample_reviews += f"📝 {row['Review'][:200]}{'...' if len(row['Review']) > 200 else ''}\n"
            sample_reviews += "-" * 60 + "\n\n"

        # Export functionality - save full dataframe
        csv_data = df.to_csv(index=False)

        return summary_html, sample_reviews, csv_data

    except Exception as e:
        error_msg = (
            f"❌ Error occurred: {str(e)}\n\nPlease check the app ID and try again."
        )
        return error_msg, "", ""


# Example reviews for single review tab
examples = [
    "This app is amazing! Love all the features and the interface is so smooth.",
    "The app crashes frequently and customer support is terrible. Very disappointed.",
    "It's okay, does what it's supposed to do. Nothing special but not bad either.",
    "Absolutely fantastic! Best app I've ever used. Highly recommend!",
    "Terrible experience. Waste of money and time. Don't download this.",
    "Pretty decent app. Some bugs here and there but overall it works fine.",
]

# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    max-width: 1200px !important;
    margin: auto !important;
}

.main-header {
    text-align: center;
    padding: 2rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

.input-box textarea {
    border-radius: 10px !important;
    border: 2px solid #e0e0e0 !important;
    font-size: 16px !important;
}

.input-box textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

.gr-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    font-weight: bold !important;
    font-size: 16px !important;
    transition: transform 0.2s !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3) !important;
}

footer {
    display: none !important;
}
"""

# Create the Gradio interface with tabs
with gr.Blocks(
    css=custom_css, theme=gr.themes.Soft(), title="App Review Sentiment Analyzer"
) as demo:

    # Main Header
    gr.HTML(
        """
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5em; font-weight: 800;">📱 App Review Sentiment Analyzer</h1>
            <p style="margin-top: 10px; font-size: 1.1em; opacity: 0.95;">
                Powered by BERT (96% Accuracy) - Analyze sentiment instantly!
            </p>
        </div>
    """
    )

    with gr.Tabs():
        # Tab 1: Single Review Analysis
        with gr.Tab("📝 Single Review Analysis"):
            gr.Markdown(
                """
                ### 🎯 How to use:
                1. Enter an app review in the text box below
                2. Click **Analyze Sentiment** or press Enter
                3. View the sentiment prediction and confidence scores
            """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    input_text = gr.Textbox(
                        label="📝 Enter App Review",
                        placeholder="Type or paste an app review here...",
                        lines=5,
                        elem_classes="input-box",
                    )
                    analyze_btn = gr.Button(
                        "🔍 Analyze Sentiment", variant="primary", size="lg"
                    )

                with gr.Column(scale=1):
                    output_html = gr.HTML(label="📊 Sentiment Result")
                    confidence_output = gr.Textbox(
                        label="📈 Confidence Distribution", lines=4, interactive=False
                    )

            gr.Markdown("### 💡 Try these examples:")

            # Create example buttons with actual text previews
            with gr.Row():
                example_buttons = []
                for i, example in enumerate(examples[:3]):
                    # Show first 30 characters of the example
                    btn_text = example[:30] + "..." if len(example) > 30 else example
                    btn = gr.Button(btn_text, size="sm")
                    example_buttons.append(btn)

            with gr.Row():
                for i, example in enumerate(examples[3:]):
                    # Show first 30 characters of the example
                    btn_text = example[:30] + "..." if len(example) > 30 else example
                    btn = gr.Button(btn_text, size="sm")
                    example_buttons.append(btn)

            # Set up example button clicks
            for i, btn in enumerate(example_buttons):
                btn.click(lambda x=examples[i]: x, outputs=input_text)

            analyze_btn.click(
                fn=predict_sentiment,
                inputs=input_text,
                outputs=[output_html, confidence_output],
            )

            input_text.submit(
                fn=predict_sentiment,
                inputs=input_text,
                outputs=[output_html, confidence_output],
            )

        # Tab 2: App Review Scraper & Analysis
        with gr.Tab("🔍 App Review Scraper & Bulk Analysis"):
            gr.Markdown(
                """
                ### 📱 Analyze Real App Reviews from Google Play Store
                
                **Easy to use - Just paste the Google Play Store URL!**
                1. Go to Google Play Store and find your app
                2. Copy the entire URL from your browser (e.g., `https://play.google.com/store/apps/details?id=com.instagram.android`)
                3. Paste it below - we'll automatically extract the app ID!
                
                **Or use the app package name directly:**
                - Instagram: `com.instagram.android`
                - WhatsApp: `com.whatsapp`
                - Spotify: `com.spotify.music`
                - TikTok: `com.zhiliaoapp.musically`
            """
            )

            with gr.Row():
                app_id_input = gr.Textbox(
                    label="🔗 Google Play Store URL or App Package Name",
                    placeholder="Paste: https://play.google.com/store/apps/details?id=com.instagram.android OR just: com.instagram.android",
                    value="com.instagram.android",
                )
                num_reviews_input = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="📊 Number of Reviews to Analyze",
                )

            scrape_btn = gr.Button(
                "🚀 Scrape & Analyze Reviews", variant="primary", size="lg"
            )

            summary_output = gr.HTML(label="📊 Analysis Summary & Visualizations")

            with gr.Row():
                with gr.Column(scale=2):
                    reviews_text_output = gr.Textbox(
                        label="📝 Sample Reviews (Preview)",
                        lines=15,
                        interactive=False,
                    )

                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ### 💾 Download Full Results
                        
                        Get all reviews with their sentiment analysis in CSV format.
                        Perfect for further analysis in Excel or other tools!
                        
                        **CSV includes:**
                        - Original review text
                        - Star rating
                        - Review date
                        - AI sentiment prediction
                        - Sentiment confidence score
                        """
                    )

                    # CSV download
                    csv_output = gr.File(
                        label="📥 Download CSV File",
                        visible=True,
                        interactive=False,
                    )

            def scrape_wrapper(app_url_or_id, num_reviews):
                summary, reviews_text, csv_data = scrape_and_analyze(
                    app_url_or_id, num_reviews
                )
                if csv_data and csv_data != "":
                    # Extract clean package name for filename
                    package_name = extract_package_name(app_url_or_id)
                    safe_package_name = (
                        package_name.replace(".", "_").replace("/", "_")
                        if package_name
                        else "app"
                    )

                    # Get current timestamp for unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        delete=False,
                        suffix=".csv",
                        newline="",
                        encoding="utf-8",
                        prefix=f"sentiment_analysis_{safe_package_name}_{timestamp}_",
                    ) as f:
                        f.write(csv_data)
                        csv_file = f.name

                    return summary, reviews_text, csv_file
                return summary, reviews_text, None

            scrape_btn.click(
                fn=scrape_wrapper,
                inputs=[app_id_input, num_reviews_input],
                outputs=[summary_output, reviews_text_output, csv_output],
            )

    # Footer
    gr.Markdown(
        """
        ---
        <div style="text-align: center; color: #666; padding: 20px;">
            <p><strong>Model Information:</strong> Fine-tuned BERT (bert-base-uncased) | Accuracy: 96%</p>
            <p>🎨 Built with Gradio | 🧠 Powered by Transformers | 🚀 Made with ❤️</p>
        </div>
    """
    )

# Launch the app
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 Launching Enhanced App Review Sentiment Analyzer...")
    print("=" * 50 + "\n")
    demo.launch(share=True, server_port=7861, show_error=True)