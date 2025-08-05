# How to Run the Sentiment Analysis Project

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models (if not already done)
```bash
python train_quick.py
```

### 3. Run the Web Application
```bash
python -m streamlit run web_app_simple.py
```

### 4. Run Demo
```bash
python demo_simple.py
```

## 📁 Project Structure

```
sentiment-analysis-project/
├── src/                          # Source code
│   ├── data_loader.py            # Dataset loading
│   ├── preprocessing.py          # Text preprocessing
│   ├── feature_extraction_simple.py  # Feature extraction (simplified)
│   ├── models.py                 # Machine learning models
│   ├── evaluation.py             # Model evaluation
│   └── visualization.py          # Data visualization
├── models/                       # Trained models (created after training)
├── data/                         # Dataset storage
├── results/                      # Evaluation results
├── plots/                        # Generated plots
├── web_app_simple.py            # Streamlit web application
├── demo_simple.py               # Quick demo script
├── train_quick.py               # Quick training script
└── requirements.txt              # Python dependencies
```

## 🎯 What Each Script Does

### `train_quick.py`
- Loads IMDB sample dataset
- Preprocesses text data
- Extracts TF-IDF and Bag-of-Words features
- Trains multiple models (Naive Bayes, SVM, Logistic Regression, Random Forest, Gradient Boosting)
- Saves trained models to `models/` directory

### `web_app_simple.py`
- Interactive web interface using Streamlit
- Loads pre-trained models
- Allows single text analysis
- Supports batch analysis
- Provides dataset analysis
- Shows visualizations

### `demo_simple.py`
- Quick demonstration of the pipeline
- Shows data loading, preprocessing, feature extraction
- Trains a simple model
- Tests on sample texts

## 🔧 Troubleshooting

### If you get import errors:
1. Make sure you're in the project directory
2. Run `pip install -r requirements.txt`
3. Check that all files are in the correct locations

### If Streamlit doesn't work:
1. Try: `python -m streamlit run web_app_simple.py`
2. Make sure models are trained first: `python train_quick.py`

### If models don't load:
1. Run the training script: `python train_quick.py`
2. Check that `models/` directory contains `.pkl` files

## 📊 Available Models

After training, you'll have these models:
- **Naive Bayes** (TF-IDF and BoW versions)
- **Support Vector Machine** (TF-IDF and BoW versions)
- **Logistic Regression** (TF-IDF and BoW versions)
- **Random Forest** (TF-IDF and BoW versions)
- **Gradient Boosting** (TF-IDF and BoW versions)

## 🌐 Web Application Features

1. **Single Text Analysis**: Enter text and get sentiment prediction
2. **Batch Analysis**: Upload CSV file or enter multiple texts
3. **Dataset Analysis**: Test models on sample datasets
4. **Model Selection**: Choose different models from sidebar
5. **Visualizations**: Interactive charts and graphs

## 📈 Performance

The models are trained on a small sample dataset for demonstration. For better performance:
- Use larger datasets
- Tune hyperparameters
- Use more sophisticated feature extraction
- Implement cross-validation

## 🎓 Learning Objectives

This project demonstrates:
- **Natural Language Processing (NLP)**: Text preprocessing, tokenization, stemming
- **Machine Learning**: Supervised learning for text classification
- **Feature Engineering**: Bag-of-Words, TF-IDF
- **Model Evaluation**: Accuracy, precision, recall, F1-score
- **Web Development**: Streamlit for interactive applications
- **Data Visualization**: Charts and graphs for analysis

## 🚀 Next Steps

1. **Explore the code**: Look at the source files in `src/`
2. **Try different texts**: Test the web app with various inputs
3. **Experiment with models**: Compare different algorithms
4. **Add new features**: Extend the functionality
5. **Improve performance**: Use larger datasets and better models

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Ensure all dependencies are installed
3. Verify that models are trained before running the web app
4. Check file paths and directory structure

Happy sentiment analyzing! 😊 