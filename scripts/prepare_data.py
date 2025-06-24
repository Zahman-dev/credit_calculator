"""
Data preparation script for German Credit Dataset
Converts raw data to properly formatted CSV with column names
"""

import pandas as pd
from pathlib import Path

def prepare_german_credit_data():
    """
    Download and prepare German Credit Dataset
    """
    # Define column names based on UCI documentation (20 attributes + class)
    column_names = [
        'Checking_account',      # Status of existing checking account
        'Duration',              # Duration in month
        'Credit_history',        # Credit history
        'Purpose',              # Purpose
        'Credit_amount',        # Credit amount
        'Savings_account',      # Savings account/bonds
        'Employment',           # Present employment since
        'Installment_rate',     # Installment rate in percentage of disposable income
        'Personal_status_sex',  # Personal status and sex
        'Other_debtors',        # Other debtors/guarantors
        'Present_residence',    # Present residence since
        'Property',             # Property
        'Age',                  # Age in years
        'Other_installment_plans', # Other installment plans
        'Housing',              # Housing
        'Existing_credits',     # Number of existing credits at this bank
        'Job',                  # Job
        'Dependents',           # Number of people being liable to provide maintenance for
        'Telephone',            # Telephone
        'Foreign_worker',       # Foreign worker
        'Risk'                  # Class (1=good, 2=bad)
    ]
    
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Check if raw data exists
    raw_data_path = data_dir / 'german.data'
    if not raw_data_path.exists():
        print("Raw german.data file not found. Please download it manually from:")
        print("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data")
        print("Save it to data/german.data")
        return False
    
    try:
        # Read the raw data (German data format is space-separated with mixed types)
        print("Reading raw German Credit data...")
        with open(raw_data_path, 'r') as f:
            lines = f.readlines()
        
        # Parse the data manually - each line should have 21 columns
        data_rows = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 21:  # Exact number of columns expected
                data_rows.append(parts)
            elif len(parts) > 21:
                # Take first 21 if there are extras
                data_rows.append(parts[:21])
            else:
                print(f"Warning: Line has {len(parts)} parts (expected 21): {line.strip()[:50]}...")
        
        df = pd.DataFrame(data_rows, columns=column_names) # type: ignore
        print(f"Parsed {len(data_rows)} rows from {len(lines)} lines")
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        print(f"Shape: {df.shape}")
        
        # Convert numeric columns
        numeric_cols = ['Duration', 'Credit_amount', 'Age', 'Installment_rate', 
                       'Present_residence', 'Existing_credits', 'Dependents', 'Risk']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic data validation
        print("\nData validation:")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Risk distribution: {df['Risk'].value_counts().to_dict()}")
        
        # Data quality checks
        issues = []
        
        # Check numeric columns (excluding Risk for min check)
        numeric_feature_cols = ['Duration', 'Credit_amount', 'Age', 'Installment_rate', 
                               'Present_residence', 'Existing_credits', 'Dependents']
        for col in numeric_feature_cols:
            if df[col].dtype not in ['int64', 'float64']:
                issues.append(f"Column {col} is not numeric: {df[col].dtype}")
            
            if df[col].min() <= 0:
                issues.append(f"Column {col} has non-positive values")
        
        # Check categorical columns
        categorical_cols = [col for col in df.columns if col not in numeric_cols + ['Risk']]
        for col in categorical_cols:
            unique_vals = df[col].nunique()
            if unique_vals > 20:  # Too many categories might indicate data issue
                issues.append(f"Column {col} has {unique_vals} unique values (might be too many)")
        
        # Check target variable
        if not df['Risk'].isin([1, 2]).all(): # type: ignore
            issues.append("Risk column contains values other than 1 or 2")
        
        if issues:
            print("Data quality issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ Data quality checks passed")
        
        # Save processed data
        output_path = data_dir / 'german_credit_data.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ Data saved to {output_path}")
        
        # Display summary statistics
        print("\nDataset Summary:")
        print(f"Total records: {len(df)}")
        print(f"Good credit (1): {(df['Risk'] == 1).sum()} ({(df['Risk'] == 1).mean():.2%})")
        print(f"Bad credit (2): {(df['Risk'] == 2).sum()} ({(df['Risk'] == 2).mean():.2%})")
        
        print("\nNumeric features summary:")
        print(df[numeric_cols].describe())
        
        print("\nCategorical features summary:")
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            print(f"{col}: {df[col].nunique()} unique values")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False

if __name__ == "__main__":
    success = prepare_german_credit_data()
    if success:
        print("\n🎉 Data preparation completed successfully!")
    else:
        print("\n❌ Data preparation failed!") 