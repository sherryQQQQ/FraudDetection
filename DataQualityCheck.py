import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
class DataQualityChecks():
    '''
    This class is used to check the data quality of the dataframe.
    '''
    def __init__(self, df):
        self.df = df
        
    def perform_data_quality_checks(self):
        print("=== Data Quality Report ===")

        print("\n1. Basic Information:")
        print(f"Number of rows: {len(self.df)}")
        print(f"Number of columns: {len(self.df.columns)}")
        
        # 2. duplicate columns
        duplicate_columns = self.df.columns[self.df.columns.duplicated()].tolist()
        if duplicate_columns:
            print(f"\n2. Duplicate columns found: {duplicate_columns}")
        else:
            print("\n2. No duplicate columns found.")
        
        # 3. na detection
        missing_data = self.df.isnull().sum()
        total_rows = len(self.df)
        missing_percentage = (missing_data / total_rows) * 100
        
        print("\n3. Missing Data Analysis:")
        missing_info = pd.DataFrame({
            'Missing Values': missing_data,
            'Percentage': missing_percentage
        })
        print(missing_info[missing_info['Missing Values'] > 0])
        
        # 4. entirely missing columns
        completely_missing = missing_data[missing_data == total_rows].index.tolist()
        if completely_missing:
            print(f"\n4. Columns with entirely missing data: {completely_missing}")
        else:
            print("\n4. No columns with entirely missing data.")
        
        self.df.drop(columns=completely_missing, inplace=True)
        
        # 5. data types
        print("\n5. Data Types:")
        print(self.df.dtypes)
        
        return {
            'duplicate_columns': duplicate_columns,
            'missing_data': missing_data,
            'completely_missing': completely_missing
        }

class DataCleaning(DataQualityChecks):
    '''
    This class is used to clean the dataframe.
    '''
    def __init__(self, df):
        super().__init__(df)
        
    def detect_outliers(self, column):
        # 1. 使用 IQR 方法检测异常值
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 2. 使用 Z-score 方法检测异常值
        z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
        
        # 3. 使用百分位数方法
        lower_percentile = self.df[column].quantile(0.01)  # 1%分位数
        upper_percentile = self.df[column].quantile(0.99)  # 99%分位数
        
        # 返回异常值信息
        outliers_iqr = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        outliers_z = self.df[z_scores > 3]
        outliers_percentile = self.df[(self.df[column] < lower_percentile) | (self.df[column] > upper_percentile)]
        
        return {
            'IQR_outliers': outliers_iqr,
            'Z_score_outliers': outliers_z,
            'Percentile_outliers': outliers_percentile,
            'bounds': {
                    'IQR': (lower_bound, upper_bound),
                    'Z_score': (-3, 3),
                    'Percentile': (lower_percentile, upper_percentile)
                }
            }
    
    def visualize_outliers(self,column):
        # 箱线图
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[column])
        plt.title(f'Boxplot of {column}')
        plt.show()
        
        # 直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=column, kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()
        
        # Q-Q图
        plt.figure(figsize=(10, 6))
        stats.probplot(self.df[column], dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {column}')
        plt.show()

    def handle_outliers(self, column, method='winsorize'):
        if method == 'winsorize':
            # Winsorization方法：将异常值替换为指定百分位数的值
            lower = self.df[column].quantile(0.01)
            upper = self.df[column].quantile(0.99)
            self.df[column] = self.df[column].clip(lower=lower, upper=upper)
            
        elif method == 'remove':
            # 直接删除异常值
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df = self.df[(self.df[column] >= lower_bound) & (self.df[column] <= upper_bound)]
            
        elif method == 'transform':
            # 使用对数转换
            self.df[column] = np.log1p(self.df[column])
            
        elif method == 'impute':
            # 使用中位数填充异常值
            median = self.df[column].median()
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df.loc[(self.df[column] < lower_bound) | (self.df[column] > upper_bound), column] = median
            
        return self.df
    

    def analyze_impact(self, column, original_stats):
        # 计算处理前后的统计量变化
        new_stats = {
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'std': self.df[column].std(),
            'skew': self.df[column].skew(),
            'kurtosis': self.df[column].kurtosis(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'count': self.df[column].count(),
            'unique_values': self.df[column].nunique()
        }
        
        # 计算变化百分比
        changes = {}
        for stat in original_stats:
            changes[stat] = (new_stats[stat] - original_stats[stat]) / original_stats[stat] * 100
        
        return changes
    
    def aggregate_functions(self, column):
        
        # 1. Get original statistics
        original_stats = {
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'std': self.df[column].std(),
            'skew': self.df[column].skew(),
            'kurtosis': self.df[column].kurtosis(),
            'min': self.df[column].min(),
            'max': self.df[column].max(),
            'count': self.df[column].count(),
            'unique_values': self.df[column].nunique()
        }
        
        # 2. Detect outliers
        outliers_info = self.detect_outliers(column)
        
        # 3. Visualize outliers
        # self.visualize_outliers(column)
        
        # 4. Handle outliers
        outliers_handled_df = self.handle_outliers(column)
        
        # 5. Calculate impact analysis
        impact_analysis = self.analyze_impact(column, original_stats)
        
        # 6. Create a comprehensive report
        report = {
            'column_name': column,
            'data_type': str(self.df[column].dtype),
            'summary_statistics': {
                'before_processing': original_stats,
                'after_processing': {
                    'mean': outliers_handled_df[column].mean(),
                    'median': outliers_handled_df[column].median(),
                    'std': outliers_handled_df[column].std(),
                    'skew': outliers_handled_df[column].skew(),
                    'kurtosis': outliers_handled_df[column].kurtosis(),
                    'min': outliers_handled_df[column].min(),
                    'max': outliers_handled_df[column].max()
                }
            },
            'outlier_analysis': {
                'IQR_method': {
                    'lower_bound': outliers_info['bounds']['IQR'][0],
                    'upper_bound': outliers_info['bounds']['IQR'][1],
                    'outlier_count': len(outliers_info['IQR_outliers']),
                    'percentage': (len(outliers_info['IQR_outliers']) / len(self.df)) * 100
                },
                'Z_score_method': {
                    'threshold': 3,
                    'outlier_count': len(outliers_info['Z_score_outliers']),
                    'percentage': (len(outliers_info['Z_score_outliers']) / len(self.df)) * 100
                },
                'percentile_method': {
                    'lower_bound': outliers_info['bounds']['Percentile'][0],
                    'upper_bound': outliers_info['bounds']['Percentile'][1],
                    'outlier_count': len(outliers_info['Percentile_outliers']),
                    'percentage': (len(outliers_info['Percentile_outliers']) / len(self.df)) * 100
                }
            },
            'impact_analysis': {
                'mean_change_percent': impact_analysis['mean'],
                'median_change_percent': impact_analysis['median'],
                'std_change_percent': impact_analysis['std'],
                'skew_change_percent': impact_analysis['skew'],
                'kurtosis_change_percent': impact_analysis['kurtosis']
            },
            'recommendations': self._generate_recommendations(outliers_info, impact_analysis)
        }
        
        return report

    def _generate_recommendations(self, outliers_info, impact_analysis):
        recommendations = []
        
        # Check if there are significant outliers
        total_outliers = max(
            len(outliers_info['IQR_outliers']),
            len(outliers_info['Z_score_outliers']),
            len(outliers_info['Percentile_outliers'])
        )
        
        if total_outliers > 0:
            outlier_percentage = (total_outliers / len(self.df)) * 100
            if outlier_percentage > 5:
                recommendations.append(f"Warning: High percentage of outliers detected ({outlier_percentage:.2f}%)")
            else:
                recommendations.append(f"Moderate number of outliers detected ({outlier_percentage:.2f}%)")
        
        # Check impact on statistics
        if abs(impact_analysis['mean']) > 10:
            recommendations.append("Significant impact on mean value after outlier treatment")
        if abs(impact_analysis['std']) > 20:
            recommendations.append("Significant impact on standard deviation after outlier treatment")
        
        # Check distribution shape
        if abs(impact_analysis['skew']) > 30:
            recommendations.append("Significant change in distribution skewness")
        if abs(impact_analysis['kurtosis']) > 40:
            recommendations.append("Significant change in distribution kurtosis")
        
        if not recommendations:
            recommendations.append("No significant issues detected")
        
        return recommendations

def missing_data_analysis(df):
    # 1. Save original fraud metrics
    original_fraud_rate = df['isFraud'].mean()
    original_fraud_count = df['isFraud'].sum()
    original_total = len(df)

    # 2. Remove missing values
    df_cleaned = df.dropna()

    # 3. Calculate new fraud metrics
    new_fraud_rate = df_cleaned['isFraud'].mean()
    new_fraud_count = df_cleaned['isFraud'].sum()
    new_total = len(df_cleaned)

    # 4. Print comparison results
    print("\n=== Fraud Analysis Before and After Removing Missing Values ===")
    print(f"\nOriginal Data:")
    print(f"- Total records: {original_total}")
    print(f"- Fraud cases: {original_fraud_count}")
    print(f"- Fraud rate: {original_fraud_rate:.4%}")

    print(f"\nAfter Removing Missing Values:")
    print(f"- Total records: {new_total}")
    print(f"- Fraud cases: {new_fraud_count}")
    print(f"- Fraud rate: {new_fraud_rate:.4%}")

    print(f"\nChanges:")
    print(f"- Records removed: {original_total - new_total} ({((original_total - new_total)/original_total):.2%})")
    print(f"- Fraud rate change: {((new_fraud_rate - original_fraud_rate)/original_fraud_rate):.2%}")

    # 5. Chi-square test for statistical significance
    from scipy.stats import chi2_contingency

    # Create contingency table
    contingency_table = pd.crosstab(
        pd.concat([df['isFraud'], df_cleaned['isFraud']], keys=['original', 'cleaned']),
        'count'
    )

    # Perform chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingency_table)

    print(f"\nStatistical Significance Test:")
    print(f"- p-value: {p_value:.4f}")
    print(f"- {'Significant' if p_value < 0.05 else 'Not significant'} difference (α=0.05)")

    # 6. Visualize comparison
    plt.figure(figsize=(10, 6))
    labels = ['Original Data', 'After Cleaning']
    fraud_rates = [original_fraud_rate, new_fraud_rate]

    plt.bar(labels, fraud_rates)
    plt.title('Fraud Rate Comparison Before and After Removing Missing Values')
    plt.ylabel('Fraud Rate')
    for i, v in enumerate(fraud_rates):
        plt.text(i, v, f'{v:.4%}', ha='center', va='bottom')
    plt.show()
