 Amazon Sales Dataset
This dataset is having the data of 1K+ Amazon Product's Ratings and Reviews as per their details listed on the official website of Amazon
Features
product_id - Product ID
product_name - Name of the Product
category - Category of the Product
discounted_price - Discounted Price of the Product
actual_price - Actual Price of the Product
discount_percentage - Percentage of Discount for the Product
rating - Rating of the Product
rating_count - Number of people who voted for the Amazon rating
about_product - Description about the Product
user_id - ID of the user who wrote review for the Product
user_name - Name of the user who wrote review for the Product
review_id - ID of the user review
review_title - Short review
review_content - Long review
img_link - Image Link of the Product
product_link - Official Website Link of the Product
Metadata
Source: This dataset is scraped from the official website of Amazon\
Collection Methodology: This dataset is scraped through BeautifulSoup and WebDriver using Python
License: CC BY-NC-SA 4.0
add Codeadd Markdown
Task
Exploring the Amazon Sales Dataset involves a step-by-step process. First, we clean and prepare the data to ensure it's accurate and consistent. Then, we summarize the data using descriptive statistics like averages and ranges. Next, we visualize the data with charts and graphs to see patterns and relationships. We detect outliers, which are unusual data points, and test our assumptions about the data. We divide the data into groups for better understanding and finally, we summarize our findings.
add Codeadd Markdown
Objectives
The primary objective of analyzing the Amazon Sales Dataset is delve into product categories, prices, ratings, and sales patterns to identify characteristics that resonate with consumers and propel them to purchase.
Delve into product categories, prices, ratings, and sales patterns to identify characteristics that resonate with consumers and propel them to purchase.
Translate insights into actionable recommendations that optimize product development, inform marketing strategies, and boost your competitive edge.
Equip businesses with the knowledge to create products that cater to evolving consumer needs and desires.
Craft communication strategies that resonate with specific demographics and maximize engagement.
Facilitate a marketplace where products find their perfect match in the hearts of consumers.
add Codeadd Markdown
Kernel Version Used
Python 3.12.0
add Codeadd Markdown
Import Libraries

We will use the following libraries
1. Pandas: Data manipulation and analysis
2. Numpy: Numerical operations and calculations
3. Matplotlib: Data visualization and plotting
4. Seaborn: Enhanced data visualization and statistical graphics
5. Scipy: Scientific computing and advanced mathematical operations
add Codeadd Markdown
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
# this is for jupyter notebook to show the plot in the notebook itself instead of opening a new window
%matplotlib inline

add Codeadd Markdown
Data Loading and Exploration | Cleaning

add Codeadd Markdown
Load a CSV file then creating a dataframe
add Codeadd Markdown
df = pd.read_csv("/kaggle/input/amazon-sales-dataset/amazon.csv")
add Codeadd Markdown
Set the option to show maximum columns
add Codeadd Markdown
pd.set_option('display.max_columns', None) 
add Codeadd Markdown
Get a sneak peek of data
The purpose of a sneak peek is to get a quick overview of the data and identify any potential problems or areas of interest.
add Codeadd Markdown
# Let's have a look on top 5 rows of the data
df.head(5)
add Codeadd Markdown
Let's see the column names
add Codeadd Markdown
df.columns
add Codeadd Markdown
Let's have a look on the shape of the dataset
add Codeadd Markdown
print(f"The Number of Rows are {df.shape[0]}, and columns are {df.shape[1]}.")
add Codeadd Markdown
Let's have a look on the columns and their data types using detailed info function
add Codeadd Markdown
df.info()
add Codeadd Markdown
df.isnull().sum()
add Codeadd Markdown
Observation Set 1

There are 1465 rows and 16 columns in the dataset.
The data type of all columns is object.
The columns in the datasets are:
'product_id', 'product_name', 'category', 'discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'about_product', 'user_id', 'user_name', 'review_id', 'review_title', 'review_content', 'img_link', 'product_link'
There are a few missing values in the dataset, which we will read in detail and deal with later on in the notebook.
add Codeadd Markdown
Changing Data Types of Columns from object to float

add Codeadd Markdown
# Changing the data type of discounted price and actual price

df['discounted_price'] = df['discounted_price'].str.replace("₹",'')
df['discounted_price'] = df['discounted_price'].str.replace(",",'')
df['discounted_price'] = df['discounted_price'].astype('float64')

df['actual_price'] = df['actual_price'].str.replace("₹",'')
df['actual_price'] = df['actual_price'].str.replace(",",'')
df['actual_price'] = df['actual_price'].astype('float64')
add Codeadd Markdown
# Changing Datatype and values in Discount Percentage

df['discount_percentage'] = df['discount_percentage'].str.replace('%','').astype('float64')

df['discount_percentage'] = df['discount_percentage'] / 100
add Codeadd Markdown
# Finding unusual string in rating column
df['rating'].value_counts()
add Codeadd Markdown
# Check the strange row
df.query('rating == "|"')
add Codeadd Markdown
I got this product rating on Amazon by searching the provided product_id on their official website (amazon.in)
The rating is 3.9. So, I am going to give the item rating a 3.9 as well.
add Codeadd Markdown
# Changing Rating Columns Data Type

df['rating'] = df['rating'].str.replace('|', '3.9').astype('float64')
add Codeadd Markdown
# Changing 'rating_count' Column Data Type

df['rating_count'] = df['rating_count'].str.replace(',', '').astype('float64')
add Codeadd Markdown
df.info()
add Codeadd Markdown
Descriptive Statistics

Descriptive statistics are a collection of quantitative measures that summarize and describe the main characteristics of a dataset.
add Codeadd Markdown
df.describe()
add Codeadd Markdown
Observation Set 2

All columns data type was object So, I converted some column data type to float.
There are 4 numeric as per Python coding or descriptive statistics from Python describe function
add Codeadd Markdown
add Codeadd Markdown
Dealing with the missing values

add Codeadd Markdown
Dealing with the missing values is one of the most important part of the data wrangling process, we must deal with the missing values in order to get the correct insights from the data.
add Codeadd Markdown
Missing Values
add Codeadd Markdown
df.isnull().sum().sort_values(ascending = False)
add Codeadd Markdown
# Find missing values percentage in the data
round(df.isnull().sum() / len(df) * 100, 2).sort_values(ascending=False) 
add Codeadd Markdown
# Find total number of missing values
df.isnull().sum().sum()
add Codeadd Markdown
Let's plot the missing values
add Codeadd Markdown
# make a figure size
plt.figure(figsize=(22, 10))
# plot the null values in each column
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis') 
add Codeadd Markdown
    Figure-1: Heatmap of Missing Values
add Codeadd Markdown
Let's plot the missing values by percentage
add Codeadd Markdown
# make figure size
plt.figure(figsize=(22, 10))
# plot the null values by their percentage in each column
missing_percentage = df.isnull().sum()/len(df)*100
missing_percentage.plot(kind='bar')
# add the labels
plt.xlabel('Columns')
plt.ylabel('Percentage')
plt.title('Percentage of Missing Values in each Column')
add Codeadd Markdown
    Figure-2: This is a percentage null values plot.
add Codeadd Markdown
We are only viewing the rows where there are null values in the column.
add Codeadd Markdown
df[df['rating_count'].isnull()].head(5)
add Codeadd Markdown
# Impute missing values
df['rating_count'] = df.rating_count.fillna(value=df['rating_count'].median())
add Codeadd Markdown
df.isnull().sum().sort_values(ascending = False)
add Codeadd Markdown
Milestone 1: We have cleaned the dataset from null values
add Codeadd Markdown
add Codeadd Markdown
Find Duplications and Analyse them

add Codeadd Markdown
Duplicates
Removing duplicates is one of the most important part of the data wrangling process, we must remove the duplicates in order to get the correct insights from the data.
If you do not remove duplicates from a dataset, it can lead to incorrect insights and analysis.
Duplicates can skew statistical measures such as mean, median, and standard deviation, and can also lead to over-representation of certain data points.
It is important to remove duplicates to ensure the accuracy and reliability of your data analysis.
add Codeadd Markdown
# Find Duplicate
df.duplicated().any()
add Codeadd Markdown
df.columns
add Codeadd Markdown
any_duplicates = df.duplicated(subset=['product_id', 'product_name', 'category', 'discounted_price',
       'actual_price', 'discount_percentage', 'rating', 'rating_count',
       'about_product', 'user_id', 'user_name', 'review_id', 'review_title',
       'review_content', 'img_link', 'product_link']).any()
add Codeadd Markdown
any_duplicates
add Codeadd Markdown
Milestone 2: Hence no duplicates found
add Codeadd Markdown
add Codeadd Markdown
Data Visualization

add Codeadd Markdown
Scatter Plot

add Codeadd Markdown
# Plot actual_price vs. rating
plt.scatter(df['actual_price'], df['rating'])
plt.xlabel('Actual_price')
plt.ylabel('Rating')
plt.show()
add Codeadd Markdown
# dont show warnings
import warnings
warnings.filterwarnings('ignore')
add Codeadd Markdown
Histogram

add Codeadd Markdown
# Plot distribution of actual_price
plt.hist(df['actual_price'])
plt.xlabel('Actual Price')
plt.ylabel('Frequency')
plt.show()
add Codeadd Markdown
from sklearn.preprocessing import LabelEncoder
# label encode categorical variables

le_product_id = LabelEncoder()
le_category = LabelEncoder()
le_review_id = LabelEncoder()
le_review_content = LabelEncoder()
le_product_name = LabelEncoder()
le_user_name = LabelEncoder()
le_about_product = LabelEncoder()
le_user_id = LabelEncoder()
le_review_title = LabelEncoder()
le_img_link = LabelEncoder()
le_product_link = LabelEncoder()


df['product_id'] = le_product_id.fit_transform(df['product_id'])
df['category'] = le_category.fit_transform(df['category'])
df['review_id'] = le_review_id.fit_transform(df['review_id'])
df['review_content'] = le_review_content.fit_transform(df['review_content'])
df['product_name'] = le_product_name.fit_transform(df['product_name'])
df['user_name'] = le_user_name.fit_transform(df['user_name'])
df['about_product'] = le_about_product.fit_transform(df['about_product'])
df['user_id'] = le_user_id.fit_transform(df['user_id'])
df['review_title'] = le_review_title.fit_transform(df['review_title'])
df['img_link'] = le_img_link.fit_transform(df['img_link'])
df['product_link'] = le_product_link.fit_transform(df['product_link'])
add Codeadd Markdown
Heatmap

add Codeadd Markdown
# Plot correlations between variables
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()
add Codeadd Markdown
Correlation Analysis:

add Codeadd Markdown
# Calculate Pearson correlation coefficients (default in Pandas)
correlation_matrix = df.corr()

# Print the correlation matrix
print(correlation_matrix)

# Create a heatmap to visualize the correlations
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Pearson)")
plt.show()

# Calculate Spearman correlation coefficients (for non-linear relationships)
spearman_correlation_matrix = df.corr(method="spearman")

# Print the Spearman correlation matrix
print(spearman_correlation_matrix)

# Create a heatmap to visualize the Spearman correlations
sns.heatmap(spearman_correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix (Spearman)")
plt.show()
add Codeadd Markdown
# Calculate correlation coefficient between product price and sales
correlation_coefficient = np.corrcoef(df['actual_price'], df['rating'])[0, 1]

# Print correlation coefficient
print(correlation_coefficient)
add Codeadd Markdown
Grouping and Aggregation

add Codeadd Markdown
# Calculate mean sales by product category
grouped_df = df.groupby('category')['rating'].mean()

# Print mean sales by product category
print(grouped_df)
add Codeadd Markdown
Calculate summary statistics for groups
add Codeadd Markdown
# Mean rating by category
mean_sales_by_category = df.groupby('category')['rating'].mean()
print(mean_sales_by_category)

# Median rating by review_content
median_sales_by_age = df.groupby('review_content')['rating'].median()
print(median_sales_by_age)

# Standard deviation of actual_price by product_name
std_price_by_brand = df.groupby('product_name')['actual_price'].std()
print(std_price_by_brand)
add Codeadd Markdown
Create pivot tables

add Codeadd Markdown
# Pivot table of rating by category and customer location
pivot_table = df.pivot_table(values='rating', index='category', columns='product_link', aggfunc='mean')
print(pivot_table)

# Pivot table of average rating_count by customer age group and product category
pivot_table = df.pivot_table(values='rating_count', index='review_content', columns='category', aggfunc='mean')
print(pivot_table)
add Codeadd Markdown
Statistical Tests:

add Codeadd Markdown
import scipy.stats as stats

# Conduct t-test to compare rating between two categories
t_statistic, p_value = stats.ttest_ind(df[df['category'] == 'electronics']['rating'], df[df['category'] == 'clothing']['rating'])

# Print t-statistic and p-value
print(t_statistic, p_value)
add Codeadd Markdown
df.info()
add Codeadd Markdown
# Chi-square test

# Create a contigency table
contigency_table = pd.crosstab(df['actual_price'], df['rating'])
contigency_table
add Codeadd Markdown
# perform chi-square test
chi2, p, dof, expected = stats.chi2_contingency(contigency_table)

# print the results
print('Chi-square statistic:', chi2)
print('p-value:', p)
print('Degrees of freedom:', dof)
print(f"Expected:\n {expected}")
add Codeadd Markdown
# inverse transform the data

df['product_id'] = le_product_id.inverse_transform(df['product_id'])
df['category'] = le_category.inverse_transform(df['category'])
df['review_id'] = le_review_id.inverse_transform(df['review_id'])
df['review_content'] = le_review_content.inverse_transform(df['review_content'])
df['product_name'] = le_product_name.inverse_transform(df['product_name'])
df['user_name'] = le_user_name.inverse_transform(df['user_name'])
df['about_product'] = le_about_product.inverse_transform(df['about_product'])
df['user_id'] = le_user_id.inverse_transform(df['user_id'])
df['review_title'] = le_review_title.inverse_transform(df['review_title'])
df['img_link'] = le_img_link.inverse_transform(df['img_link'])
df['product_link'] = le_product_link.inverse_transform(df['product_link'])
add Codeadd Markdown
add Codeadd Markdown
Questions and Answers

These are some questions are follows:
Q1: What is the average rating for each product category?
Q2: What are the top rating_count products by category?
Q3: What is the distribution of discounted prices vs. actual prices?
Q4: How does the average discount percentage vary across categories?
Q5: What are the most popular product name?
Q6: What are the most popular product keywords?
Q7: What are the most popular product reviews?
Q8: What is the correlation between discounted_price and rating?
Q9: What are the Top 5 categories based with highest ratings?
add Codeadd Markdown
Q1: What is the average rating for each product category?
add Codeadd Markdown
import pandas as pd

# Check the data type of the "rating" column
print(df["rating"].dtype)

# If the data type is not numeric, convert it to numeric
if df["rating"].dtype == "object":
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")  # Handle potential errors

# Calculate the average ratings after ensuring numeric data type
average_ratings = df.groupby("category")["rating"].mean().reset_index()

print(average_ratings)
add Codeadd Markdown
Answer 1:
The output shows that most product categories have generally positive customer feedback, with average ratings above 3.50. However, some categories (e.g., 2 and 3) have lower ratings, suggesting potential areas for improvement. Further analysis of these categories could help identify specific reasons for lower feedback and identify potential solutions.
add Codeadd Markdown
Q2: What are the top rating_count products by category?
add Codeadd Markdown
import pandas as pd

top_reviewed_per_category = (
    df.groupby("category")
    .apply(lambda x: x.nlargest(10, "rating_count"))
    .reset_index(drop=True)
)

print(top_reviewed_per_category)
add Codeadd Markdown
Answer 2:
The output highlights products likely to be popular within their categories based on high review counts, suggesting customer interest and engagement.
Review counts range from 9 to 15867, implying varying levels of attention and feedback across products.
Most listed products have ratings above 3.5, indicating a generally positive customer experience.
Products with the highest review counts within their categories might be considered potential top sellers, even without direct sales data.
add Codeadd Markdown
Q3: What is the distribution of discounted prices vs. actual prices?
add Codeadd Markdown
import pandas as pd

# Create histograms
df["discounted_price"].hist(label="Discounted Price")
df["actual_price"].hist(label="Actual Price")

# Calculate and analyze discount percentages
df["discount_percentage"] = (df["actual_price"] - df["discounted_price"]) / df["actual_price"] * 100
df["discount_percentage"].describe()
df["discount_percentage"].hist(label="Discount Percentage")

add Codeadd Markdown
Answer 3:
The output shows that discounted prices are generally lower than actual prices, with a median discounted price of $200 and a median actual price of $400.
The discount percentage distribution is skewed to the left, with most products having a discount of 30% or less.
The output suggests that there may be opportunities to increase discounted prices or discount percentages to attract more customers.
add Codeadd Markdown
Q4: How does the average discount percentage vary across categories?
add Codeadd Markdown
# Calculate average discount percentage per category
avg_discount_per_category = df.groupby('category')['discount_percentage'].mean()

# Display results
print(avg_discount_per_category)

# Optional: Visualization
sns.barplot(x=avg_discount_per_category.index, y=avg_discount_per_category.values)
plt.xlabel("Category")
plt.ylabel("Average Discount Percentage")
plt.show()
add Codeadd Markdown
Answer 4:
Average discount percentages vary widely across categories, ranging from 0% to 78.39%.
Categories 1 and 3 stand out with notably higher average discounts (78.39% and 56.34%), suggesting potential factors like clearance efforts, high competition, or lower-profit margins.
Categories 0, 206, 207, 210 have average discounts of 0%, indicating consistent pricing or strong demand for products within those categories.
Other categories exhibit varying discount percentages, likely reflecting diverse pricing strategies and market dynamics.
add Codeadd Markdown
Q5: What are the most popular product name?
add Codeadd Markdown
# Count occurrences of product names
product_counts = df["product_name"].value_counts()

# Sort in descending order and display top results
print(product_counts.sort_values(ascending=False).head(10))
add Codeadd Markdown
Answer 5:
Fire-Boltt Ninja Call Pro Plus Smart Watch is the most popular product, followed by Fire-Boltt Phoenix Smart Watch.
Smart Watches and Charging Cables are the most popular product categories.
Multiple brands are represented, with boAt appearing twice.
Fast charging, durability, and functionality are key features.
Popularity is relatively evenly distributed beyond the leading product.
add Codeadd Markdown
Q6: What are the most popular product keywords?
add Codeadd Markdown
def extract_keywords(product_name):
  """Extracts keywords from a product name, handling potential numbers."""
  if isinstance(product_name, str):  # Check if it's a string
    keywords = product_name.lower().split()  # Split into words and lowercase
    keywords = [word for word in keywords if word.isalpha()]  # Remove non-alphabetical characters
  else:
    keywords = []  # Handle non-string values (e.g., integers) by returning an empty list
  return keywords

# Apply the function to extract keywords
df["keywords"] = df["product_name"].apply(extract_keywords)

# Flatten the list of keywords
all_keywords = [keyword for keywords in df["keywords"] for keyword in keywords]

# Count keyword occurrences
keyword_counts = pd.Series(all_keywords).value_counts()

# Display the top 10 most popular keywords
print(keyword_counts.head(10))
add Codeadd Markdown
Answer 6:
USB connectivity, charging (especially fast charging), and cables are prominent product features.
Prepositions and conjunctions like "with", "for", "and", "to" suggest a focus on explaining product compatibility and usage scenarios.
Cables and smart devices are likely well-represented in the dataset.
Product names tend to be concise and use common words, potentially benefiting from refined keyword extraction techniques.
add Codeadd Markdown
Q7: What are the most popular product reviews?
add Codeadd Markdown
from textblob import TextBlob  # Import TextBlob library
# Select review column
df[["product_id", "user_id", "review_content"]]

# Calculate sentiment score for each review
df["sentiment"] = df["review_content"].apply(lambda text: TextBlob(text).sentiment.polarity)

# Sort by sentiment score (ascending for positive)
df_sorted = df.sort_values(by="sentiment", ascending=True)

# Display top reviews based on a desired number (e.g., top 10)
top_reviews = df_sorted.head(10)
print(top_reviews)
add Codeadd Markdown
Answer 7:
The overall sentiment scores are relatively low, suggesting a tendency towards neutral or slightly negative reviews in the sample.

The review with the highest sentiment score is "I have installed this in my kitchen working fine" (product_id 1463) with a score of -0.170167, indicating a mildly positive sentiment.

The review with the lowest sentiment score is "tv on off not working, so difficult to battery charge" (product_id 155) with a score of -0.600000, suggesting a strongly negative sentiment.

Several reviews mention issues with battery charging (product_id 155), product quality (product_id 1237), and ease of use (product_id 1198), highlighting potential areas for improvement.

Some reviews express both positive and negative aspects within the same text, like "Like and happy,,Please don't buy this heater" (product_id 1237), suggesting a nuanced evaluation of the product.

The user_id column seems to contain commas, indicating multiple user IDs for some reviews. This might need investigation to ensure accuracy.

Reviews for product_id 22, 152, and 723 have identical content, suggesting potential data duplication or errors.

add Codeadd Markdown
Q8: What is the correlation between discounted_price and rating?
add Codeadd Markdown
# Calculate the correlation coefficient
correlation_coefficient = df["discounted_price"].corr(df["rating"])

# Print the correlation coefficient with two decimal places
print(f"Correlation between discounted_price and rating: {correlation_coefficient:.2f}")
add Codeadd Markdown
Answer 8:
Discounted price and rating have a weak positive correlation. This means that products with higher discounted prices tend to have slightly higher ratings, but the relationship is not very strong.

add Codeadd Markdown
Q9: What are the Top 5 categories based with highest ratings?
add Codeadd Markdown
# Group data by category and calculate average rating
average_ratings = df.groupby("category")["rating"].mean().reset_index()

# Sort by average rating in descending order
average_ratings = average_ratings.sort_values(by="rating", ascending=False)

# Print the top 5 categories
print("Top 5 categories with highest average ratings:")
for i in range(5):
    category = average_ratings.iloc[i]["category"]
    average_rating = average_ratings.iloc[i]["rating"]
    print(f"{i+1}. {category}: {average_rating:.2f}")
add Codeadd Markdown
Answer 9:
The top 5 categories have average ratings between 4.50 and 4.60, indicating overall positive customer satisfaction within these areas.
Most of the top-rated categories fall within technology-related domains, including tablets, networking devices, photography accessories, media streaming devices, and calculators.
Within broader categories like "Computers & Accessories" and "Electronics," specific subcategories emerge as particularly well-rated, such as tablets, powerline adapters, film accessories, and streaming clients.
Four categories share a rating of 4.50, suggesting similar levels of customer satisfaction across these areas.
The presence of "Basic Calculators" in the top 5 suggests that even relatively simple products can achieve high ratings if they meet customer needs effectively.
add Codeadd Markdown
Summary

Our insightful exploration of the Amazon Sales dataset, characterized by its remarkable cleanliness and consistency, yielded a treasure trove of findings. Through a series of targeted inquiries, we unlocked detailed answers and shed light on previously veiled aspects of the data and findings as follows:
add Codeadd Markdown
Q1: What is the average rating for each product category?
Answer 1:
The output shows that most product categories have generally positive customer feedback, with average ratings above 3.50. However, some categories (e.g., 2 and 3) have lower ratings, suggesting potential areas for improvement. Further analysis of these categories could help identify specific reasons for lower feedback and identify potential solutions.
Q2: What are the top rating_count products by category?
Answer 2:
- The output highlights products likely to be popular within their categories based on high review counts, suggesting customer interest and engagement.
- Review counts range from 9 to 15867, implying varying levels of attention and feedback across products.
- Most listed products have ratings above 3.5, indicating a generally positive customer experience.
- Products with the highest review counts within their categories might be considered potential top sellers, even without direct sales data.
Q3: What is the distribution of discounted prices vs. actual prices?
Answer 3:
- The output shows that discounted prices are generally lower than actual prices, with a median discounted price of $200 and a median actual price of $400.
- The discount percentage distribution is skewed to the left, with most products having a discount of 30% or less.
- The output suggests that there may be opportunities to increase discounted prices or discount percentages to attract more customers.
Q4: How does the average discount percentage vary across categories?
Answer 4:
- Average discount percentages vary widely across categories, ranging from 0% to 78.39%.
- Categories 1 and 3 stand out with notably higher average discounts (78.39% and 56.34%), suggesting potential factors like clearance efforts, high competition, or lower-profit margins.
- Categories 0, 206, 207, 210 have average discounts of 0%, indicating consistent pricing or strong demand for products within those categories.
- Other categories exhibit varying discount percentages, likely reflecting diverse pricing strategies and market dynamics.
Q5: What are the most popular product name?
Answer 5:
- Fire-Boltt Ninja Call Pro Plus Smart Watch is the most popular product, followed by Fire-Boltt Phoenix Smart Watch.
- Smart Watches and Charging Cables are the most popular product categories.
- Multiple brands are represented, with boAt appearing twice.
- Fast charging, durability, and functionality are key features.
- Popularity is relatively evenly distributed beyond the leading product.
Q6: What are the most popular product keywords?
Answer 6:
- USB connectivity, charging (especially fast charging), and cables are prominent product features.
- Prepositions and conjunctions like "with", "for", "and", "to" suggest a focus on explaining product compatibility and usage scenarios.
- Cables and smart devices are likely well-represented in the dataset.
- Product names tend to be concise and use common words, potentially benefiting from refined keyword extraction techniques.
Q7: What are the most popular product reviews?
Answer 7:
- The overall sentiment scores are relatively low, suggesting a tendency towards neutral or slightly negative reviews in the sample.
- The review with the highest sentiment score is "I have installed this in my kitchen working fine" (product_id 1463) with a score of -0.170167, indicating a mildly positive sentiment.
- The review with the lowest sentiment score is "tv on off not working, so difficult to battery charge" (product_id 155) with a score of -0.600000, suggesting a strongly negative sentiment.
- Several reviews mention issues with battery charging (product_id 155), product quality (product_id 1237), and ease of use (product_id 1198), highlighting potential areas for improvement.
- Some reviews express both positive and negative aspects within the same text, like "Like and happy,,Please don't buy this heater" (product_id 1237), suggesting a nuanced evaluation of the product.
- The user_id column seems to contain commas, indicating multiple user IDs for some reviews. This might need investigation to ensure accuracy.
- Reviews for product_id 22, 152, and 723 have identical content, suggesting potential data duplication or errors.
Q8: What is the correlation between discounted_price and rating?
Answer 8:
Discounted price and rating have a weak positive correlation. This means that products with higher discounted prices tend to have slightly higher ratings, but the relationship is not very strong.
Q9: What are the Top 5 categories based with highest ratings?
Answer 9:
- The top 5 categories have average ratings between 4.50 and 4.60, indicating overall positive customer satisfaction within these areas.
- Most of the top-rated categories fall within technology-related domains, including tablets, networking devices, photography accessories, media streaming devices, and calculators.
- Within broader categories like "Computers & Accessories" and "Electronics," specific subcategories emerge as particularly well-rated, such as tablets, powerline adapters, film accessories, and streaming clients.
- Four categories share a rating of 4.50, suggesting similar levels of customer satisfaction across these areas.
- The presence of "Basic Calculators" in the top 5 suggests that even relatively simple products can achieve high ratings if they meet customer needs effectively.
add Codeadd Markdown
add Codeadd Markdown
Conclusion

The primary goal of this project is to analyze the Amazon Sales dataset and identify insights based on the data. The Amazon Sales dataset is a valuable resource for businesses and researchers alike. It provides a wealth of information about customer behavior, product trends, and market conditions. By conducting exploratory data analysis (EDA) on this dataset, businesses can gain valuable insights that can help them make better decisions about their products, marketing, and operations.
add Codeadd Markdown
During this EDA exercise, we have achieved several milestones:
- We have cleaned the dataset from null values.
- No duplications found in the data.
add Codeadd Markdown
Contact Details

Click on link below to contact/follow/correct me:
LinkedIn
Facebook
Twitter
Kaggle
Medium
add Codeadd Markdown
add Codeadd Markdown
