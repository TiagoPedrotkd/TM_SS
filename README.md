# Text Mining Project 2025

## Project Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Project Structure

```
text_mining2425_ims/
├── data/               # Data files
│   ├── train.csv      # Training data
│   └── test.csv       # Test data
├── notebooks/          # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── src/               # Source code
├── models/            # Saved models
├── results/           # Results and outputs
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Getting Started

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/01_data_exploration.ipynb` to begin exploring the data.

## Project Progress

- [ ] Data Exploration
- [ ] Data Preprocessing
- [ ] Feature Engineering
- [ ] Model Development
- [ ] Model Evaluation
- [ ] Final Submission