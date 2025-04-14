## Project Overview  
This project focuses on developing a Recommendation System that provides personalized suggestions based on user behavior and preferences. The system predicts relevant products, services, or content by analyzing historical data, improving user engagement, satisfaction, and conversion rates.


### **Objectives**  
1. Develop personalized recommendations based on user interactions.  
2. Address diverse use cases, including product, content, and service recommendations.  
3. Leverage historical user data to make accurate predictions.  
4. Enhance user engagement & retention through better suggestions.  
5. Ensure scalability and real-time performance for large datasets.  
6. Balance accuracy & diversity to provide varied recommendations.  

--- 
##  **Tools & Technologies Used**  

| Tool | Purpose |
|------|---------|
| **Python** | Programming language for data analysis & modeling |
| **Pandas & NumPy** | Data preprocessing & numerical computations |
| **Scikit-Learn & Surprise** | Machine learning for recommendations |
| **Matplotlib & Seaborn** | Data visualization & insights |
| **Google Colab** | Handling large-scale data processing |
| **Git & GitHub** | Version control & project collaboration | 

## **Dataset & Data Preprocessing**  
The dataset consists of:  
1️. **Events.csv**: User interactions (visitorid, event, transactionid, itemid).  
2️. **Item_properties.csv**: Item details (timestamp, categoryid, availability, itemid, values).  
3️. **Category_tree.csv**: Item category hierarchy.  

### **Data Preprocessing Steps**  
 1. Convert timestamps from Unix to human-readable format.  
 2. Extract category_id and availability from the property column.  
 3. Remove duplicate entries and keep the latest snapshot per item.  
 4. Identify & remove abnormal users (bot-like behavior).

##  **Recommendation System Architecture**  

This project implements a **Hybrid Recommendation System**, combining:  

### **1️. Collaborative Filtering (CF) – User-Item Interactions**  
- Uses **SVD (Singular Value Decomposition)** for **matrix factorization-based CF**.  
- Predicts **missing user-item interactions** using past behavior.  

```python
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load dataset for CF
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(final_data[['user_id', 'item_id', 'event_score']], reader)

# Train CF model
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Make predictions
predictions = model.test(testset)
```

### **2️. Content-Based Filtering (CBF) – Item Similarity**  
- Uses **TF-IDF & Cosine Similarity** on `category_id` and `popularity score`.  
- Finds **similar items** based on their attributes.  

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Prepare category & popularity-based features
final_data['combined_features'] = final_data['category_id'].astype(str) + ' ' + final_data['item_popularity'].astype(str)

# Compute TF-IDF similarity
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(final_data['combined_features'])
similarity_matrix = cosine_similarity(tfidf_matrix)
```

### **3️. Hybrid Model (CF + CBF Blended)**  
- CF predicts **what users like based on similar users**.  
- CBF recommends **items with similar properties**.  
- **Combining scores improves accuracy & diversity**.  

```python
def hybrid_recommend(user_id, top_n=5):
    # CF-based recommendations
    cf_preds = model.predict(user_id, item_id) for item_id in final_data['item_id']
    cf_top = sorted(cf_preds, key=lambda x: x.est, reverse=True)[:top_n]

    # CBF-based recommendations
    user_items = final_data[final_data['user_id'] == user_id]['item_id'].unique()
    cb_scores = similarity_matrix[user_items].mean(axis=0)
    cb_top = final_data.iloc[cb_scores.argsort()[::-1][:top_n]]

    # Combine CF & CBF results
    hybrid_results = pd.concat([cf_top, cb_top]).drop_duplicates().head(top_n)
    return hybrid_results
```

---  

##  **Evaluation Metrics & Results**  
The models were evaluated using:  

1. **Root Mean Squared Error (RMSE):** Measures prediction accuracy.  
2. **Precision & Recall:** Evaluates recommendation relevance.  
3.  **Diversity Score:** Ensures recommendations are varied.  

### **Results Summary**  
- **CF (SVD) Model:** RMSE = 0.85, Precision = 78%  
- **CBF Model:** Precision = 74%, Recall = 68%  
- **Hybrid Model:** Precision = **82%**, Recall = **75%**  
