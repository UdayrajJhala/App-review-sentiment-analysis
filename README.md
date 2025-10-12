# 📱 App Review Sentiment Analyzer

A powerful sentiment analysis tool powered by BERT (96% accuracy) that analyzes app reviews from Google Play Store. Built with Gradio for an intuitive web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![BERT](https://img.shields.io/badge/BERT-base--uncased-green.svg)
![Gradio](https://img.shields.io/badge/gradio-4.0+-orange.svg)
![Accuracy](https://img.shields.io/badge/accuracy-96%25-brightgreen.svg)

## ✨ Features

### 🎯 Single Review Analysis

- Analyze individual app reviews instantly
- Get sentiment predictions with confidence scores
- Visual sentiment representation with emojis and colors
- Try predefined examples with actual review text

### 🔍 Bulk Review Analysis

- **Easy URL Input**: Just paste the complete Google Play Store URL!
- Automatically extracts app package name from URLs
- Scrape and analyze up to 200 reviews at once
- Beautiful visualizations with sentiment distribution charts
- Export results to CSV for further analysis
- Preview sample reviews with full sentiment breakdown

### 📊 Advanced Analytics

- Sentiment distribution pie charts
- Sentiment analysis by star rating
- Detailed confidence scores for each prediction
- Comprehensive summary with app information

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone or download the project**

   ```bash
   git clone <repository-url>
   cd "App review sentiment analysis"
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements_ui.txt
   ```

3. **Run the application**

   ```bash
   python app_sentiment_ui_fixed.py
   ```

4. **Open your browser**
   - The app will automatically launch at `http://localhost:7861`
   - Or use the provided public URL for sharing

## 📦 Dependencies

```
gradio>=4.0.0          # Web interface
torch>=2.0.0           # PyTorch for deep learning
transformers>=4.30.0   # Hugging Face transformers for BERT
numpy>=1.24.0          # Numerical operations
pandas>=2.0.0          # Data manipulation
matplotlib>=3.7.0      # Plotting and visualizations
google-play-scraper>=3.1.2  # Google Play Store scraping
```

## 🎮 How to Use

### Single Review Analysis

1. Navigate to the "📝 Single Review Analysis" tab
2. Enter an app review in the text box
3. Click "🔍 Analyze Sentiment" or press Enter
4. View the sentiment prediction with confidence scores
5. Try the example buttons to see different sentiment types

### Bulk App Analysis

1. Go to the "🔍 App Review Scraper & Bulk Analysis" tab
2. **Easy Method**: Paste the complete Google Play Store URL
   ```
   Example: https://play.google.com/store/apps/details?id=com.instagram.android
   ```
3. **Alternative**: Enter the app package name directly
   ```
   Example: com.instagram.android
   ```
4. Set the number of reviews to analyze (10-200)
5. Click "🚀 Scrape & Analyze Reviews"
6. View the comprehensive analysis and download CSV results

### Finding App Package Names

- **Instagram**: `com.instagram.android`
- **WhatsApp**: `com.whatsapp`
- **Spotify**: `com.spotify.music`
- **TikTok**: `com.zhiliaoapp.musically`
- **YouTube**: `com.google.android.youtube`

## 🧠 Model Information

- **Architecture**: Fine-tuned BERT (bert-base-uncased)
- **Accuracy**: 96% on validation set
- **Classes**: 3 sentiment classes (Positive, Neutral, Negative)
- **Input**: Text reviews up to 128 tokens
- **Output**: Sentiment prediction with confidence scores

## 📁 Project Structure

```
App review sentiment analysis/
├── app_sentiment_ui_fixed.py      # Main Gradio application
├── requirements_ui.txt            # Python dependencies
├── user_reviews.csv              # Sample review data
├── README.md                     # This file
├── app_review_sentiment_bert/    # Trained BERT model
│   ├── config.json
│   ├── model.safetensors
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── flagged/                      # Gradio logs
    └── log.csv
```

## 🎨 Features Highlight

### Smart URL Processing

The app automatically detects and extracts package names from Google Play Store URLs:

- Full URLs: `https://play.google.com/store/apps/details?id=com.example.app`
- Direct package names: `com.example.app`

### Rich Visualizations

- **Sentiment Distribution**: Interactive pie charts with emoji labels
- **Star Rating Analysis**: Bar charts showing sentiment by rating
- **Color-coded Results**: Intuitive color scheme (Green=Positive, Yellow=Neutral, Red=Negative)

### Export Capabilities

Download comprehensive CSV reports containing:

- Original review text
- Star ratings
- Review dates
- AI sentiment predictions
- Confidence scores

## 🔧 Customization

### Adjusting Review Count

Modify the slider range in the code:

```python
num_reviews_input = gr.Slider(
    minimum=10,
    maximum=500,  # Increase for more reviews
    value=50,
    step=10,
    label="📊 Number of Reviews to Analyze",
)
```

### Adding New Examples

Update the examples list:

```python
examples = [
    "Your new example review here...",
    # Add more examples
]
```

## 🚨 Troubleshooting

### Common Issues

1. **App ID not found**

   - Verify the Google Play Store URL is correct
   - Ensure the app is available in the US store
   - Try using the package name directly

2. **Model loading errors**

   - Check if the `app_review_sentiment_bert` folder exists
   - Ensure all model files are present
   - Verify PyTorch installation

3. **Memory issues**
   - Reduce the number of reviews to analyze
   - Use CPU instead of GPU if memory is limited

### Performance Tips

- Use GPU for faster inference (automatic detection)
- Limit review count for faster processing
- Close other applications to free up memory

## 📈 Model Performance

- **Training Data**: Curated app review dataset
- **Validation Accuracy**: 96%
- **F1-Score**: 0.94 (average across all classes)
- **Inference Speed**: ~50ms per review (GPU), ~200ms (CPU)

