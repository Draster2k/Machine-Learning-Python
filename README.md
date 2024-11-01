# Housing Price Prediction using Linear Regression

This project demonstrates a basic linear regression model to predict housing prices based on various factors such as location, population, and median income. The dataset used is `Housing.csv`, and the code is implemented in Python.

## Project Structure
- **main.py**: Python script containing the data processing, visualization, and model training.
- **Housing.csv**: Dataset used for model training and testing.
- **README.md**: Documentation of the project.

## Dataset
The `Housing.csv` dataset includes features like:
- `longitude` and `latitude`: Geographic coordinates.
- `housing_median_age`: Age of the housing in the area.
- `total_rooms` and `total_bedrooms`: Counts of rooms and bedrooms.
- `population` and `households`: Population and household counts.
- `median_income`: Median income in the area.
- `median_house_value`: Target variable for prediction.
- `ocean_proximity`: Proximity to the ocean (categorical feature).

## Methodology
1. **Data Preparation**:
   - Stratified sampling is used to ensure consistent income distribution in training and test sets.
   - One-hot encoding converts the categorical `ocean_proximity` feature to numeric.
   - Missing values in `total_bedrooms` are filled with the median for accurate model training.
   
2. **Data Visualization**:
   - A scatter plot visualizes housing data distribution by population and geographic location.

3. **Model Training**:
   - We use **linear regression** to model the relationship between the features and the target (`median_house_value`).
   - Predictions are generated on a sample to compare with actual values and evaluate model performance.

## Results
Sample predictions vs. actual values:
- **Predictions**: [88983.15, 305351.35, 153334.71, 184302.55, 246840.19]
- **Actual values**: [72100., 279600., 82700., 112500., 238300.]

## How to Run
1. Clone this repository.
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the script:
    ```bash
    python main.py
    ```
