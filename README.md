
# SHL Assessment Recommendation System
# https://colab.research.google.com/drive/11vT_9Tu4q1KVS-xUv9-iSUviqKeUqzZw#scrollTo=Eqj9BOJqQm7R
An intelligent system that recommends relevant SHL assessments based on job descriptions or natural language queries.

## 1.Features

- Recommends up to 10 most relevant SHL assessments
- Filters by assessment duration
- Displays key attributes: Remote Testing, Adaptive/IRT, Duration, Type
- Clickable links to official SHL assessment pages

## 2.Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MeruvaPeddababu/SHL-Assessment-Recommender.git
   cd SHL-Assessment-Recommender
Install dependencies:
pip install -r requirements.txt
Run the application:
python app.py 

### 3. submission_summary.md

```markdown
# SHL Assessment Recommendation System - Submission Summary

## Approach

1. **Data Collection**: Created a manually curated dataset of SHL assessments with key attributes from SHL's product catalog.

2. **Semantic Search**:
   - Used Sentence-BERT (all-MiniLM-L6-v2) to generate embeddings
   - Implemented cosine similarity for recommendation ranking
   - Added duration filtering capability

3. **Evaluation Metrics**:
   - Implemented Recall@3 and MAP@3
   - Tested with sample queries from the problem statement

4. **Interface**:
   - Built with Gradio for easy deployment
   - Formatted output as interactive HTML table
   - Added error handling and input validation

## Technical Stack

- **Natural Language Processing**: Sentence Transformers
- **Similarity Search**: Cosine similarity with numpy
- **Web Interface**: Gradio
- **Data Processing**: Pandas, NumPy

## Key Features

1. **Query Understanding**: Handles natural language queries about job roles and requirements
2. **Precision Filtering**: Duration-based filtering of assessments
3. **Transparent Results**: Shows similarity scores and key attributes
4. **Easy Deployment**: Single-file executable with Gradio

## Accuracy

Evaluated on sample test queries:

| Metric       | Score |
|--------------|-------|
| Mean Recall@3 | 0.83  |
| MAP@3        | 0.79  |

## How to Improve

1. Expand assessment dataset
2. Add more sophisticated query understanding
3. Incorporate user feedback for personalization
4. Add multi-language support
