---
language: en
license: cc-by-nc-4.0
datasets:
- reddit
task_categories:
- text-classification
- text-generation
task_ids:
- sentiment-classification
- text-regression
pretty_name: Service Industry Tipping Dataset
tags:
- service-industry
- tipping
- reddit
- restaurant
- customer-service
- tip-prediction
---

# Service Industry Tipping Dataset (Reddit)

## Dataset Description
This dataset contains tipping-related posts collected from Reddit, specifically focusing on the service industry (restaurants, cafes, bars, hotels, etc.). The data has been carefully processed, analyzed, and verified using multiple AI models to create a high-quality dataset for training tip prediction models. Each entry provides insights into service situations and corresponding tip percentages, enabling researchers and developers to explore the relationship between service quality and customer tipping behavior.

## Collection Methodology
1. **Data Sources**: Posts were collected from multiple service-industry related subreddits including r/TalesFromYourServer, r/serverlife, r/bartenders, r/KitchenConfidential, and others.
2. **Search Criteria**: Posts were identified using keywords related to tipping such as "tip", "tipping", "gratuity", "15%", "20%", etc.
3. **Relevance Scoring**: Each post was assigned a relevance score based on keyword matching and tip-related content.
4. **AI Analysis**: Tip percentages and situation descriptions were systematically extracted using Ollama and further verified with Google's Gemini AI model.

## Dataset Versions
This dataset is provided in 4 different versions to support various research needs:

1. **raw**: Original Reddit Data
   - Raw collected data without filtering
   - Contains all originally collected posts, including those that may not have clear tip information
   - Useful for developing improved extraction algorithms or for broader service industry analysis

2. **analyzed**: Data Analyzed with Ollama
   - Contains tip percentages and situation descriptions extracted from text using the Ollama AI model
   - Includes posts where the AI was able to identify tip-related information
   - Provides both structured (tip percentage/amount) and unstructured (situation description) data

3. **filtered**: Primary Filtered Data
   - Quality-controlled subset focusing on reasonable tip values
   - Includes only posts with tip percentages up to 30%
   - Tips between 20% and 30% have been normalized to 20% to reduce outlier effects
   - Filtered based on outlier detection to ensure data consistency

4. **verified**: Secondary Verified Data with Gemini
   - Gold-standard dataset with the highest quality entries
   - Further verified by Google Gemini model to ensure consistency between situation descriptions and tip percentages
   - Each entry has been evaluated for clear captions and logical tip values
   - Optimal for training accurate tip prediction models

## Data Fields
The dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique post ID from Reddit |
| subreddit | string | Subreddit where the post was published |
| title | string | Original post title |
| text_content | string | Full post body content |
| url | string | Direct link to the original Reddit post |
| score | integer | Reddit score (upvotes - downvotes) |
| num_comments | integer | Number of comments on the post |
| created_utc | integer | Post creation timestamp (Unix format) |
| relevance_score | float | Calculated relevance score for tip-related content |
| search_keyword | string | Keyword used to find this post |
| sort_method | string | Reddit sorting method used during collection |
| tip_percentage | float | Extracted tip percentage value |
| tip_amount | float | Extracted tip amount in dollars (when available) |
| situation_caption | string | AI-generated description of the service situation |
| outlier_detection | string | Indicator of whether the entry passed outlier checks ("Yes"/"No") |

## Data Statistics
- **Raw Dataset**: Original collection from multiple Reddit sources
- **Analyzed Dataset**: Posts with successfully extracted tip information
- **Filtered Dataset**: Quality-controlled subset with reasonable tip values (≤30%)
- **Verified Dataset**: Gold-standard dataset with verified consistency between descriptions and tip values

## Use Cases
This dataset is specifically designed for the following applications:

- **Tip Prediction Models**: Training AI models to predict appropriate tip percentages based on service quality descriptions
- **Service Quality Analysis**: Understanding what service aspects influence tipping behavior
- **Customer Behavior Research**: Analyzing patterns in how customers respond to different service experiences
- **Natural Language Processing**: Developing models that can extract financial information from text
- **Text-based Monetary Value Prediction**: Building systems that can suggest appropriate monetary values based on textual descriptions

## Example Usage

```python
from datasets import load_dataset

# Load the verified dataset (best quality)
dataset = load_dataset("kfkas/service-tipping-reddit-data-verified")

# Explore the data
print(f"Dataset size: {len(dataset['train'])} examples")

# Display a few examples
for example in dataset['train'][:3]:
    print(f"Situation: {example['situation_caption']}")
    print(f"Tip percentage: {example['tip_percentage']}%")
    print("-" * 50)

# Use for training a model
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=1)

# Prepare data for regression task (predicting tip percentage)
def preprocess_function(examples):
    return tokenizer(examples["situation_caption"], truncation=True)

tokenized_dataset = dataset.map(preprocess_function, batched=True)
```

## Ethical Considerations
- All data has been collected from public Reddit posts
- Personal identifying information is not included in the dataset
- The dataset is intended for research and educational purposes only
- When using this dataset, please consider ethical implications related to service industry labor practices and compensation

## License
This dataset is provided for research and educational purposes under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

## Citation
If you use this dataset in your research or applications, please cite it as follows:
```
@dataset{service_tipping_reddit_data,
  author       = {kfkas},
  title        = {Service Industry Tipping Reddit Data},
  year         = {2025},
  howpublished = {https://huggingface.co/datasets/kfkas/service-tipping-reddit-data-verified},
}
```

## Contact
For questions or feedback regarding this dataset, please contact the dataset maintainer through Huggingface.
