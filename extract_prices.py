"""
자산명 추출 및 가격 데이터 파싱
"""
import pandas as pd
import numpy as np

file_path = "가격 데이터.csv"

# 첫 번째 줄에서 자산명 추출
df_assets = pd.read_csv(file_path, encoding='cp949', nrows=1, low_memory=False)

print("="*80)
print("Asset Names Extraction")
print("="*80)

# 자산명은 8개 컬럼마다 반복
# 컬럼 인덱스: 0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80
asset_names = []
asset_indices = []

for i in range(0, len(df_assets.columns), 8):
    if i < len(df_assets.columns):
        # 첫 번째 행의 값이 자산명
        asset_name = df_assets.iloc[0, i]
        if pd.notna(asset_name) and asset_name != '':
            asset_names.append(asset_name)
            asset_indices.append(i)
            print(f"  {len(asset_names)}. Column {i}: {asset_name}")

print(f"\nTotal assets found: {len(asset_names)}")

# 실제 데이터 읽기 (skiprows=2)
print("\n" + "="*80)
print("Extracting Close Prices")
print("="*80)

df = pd.read_csv(file_path, encoding='cp949', skiprows=2, low_memory=False)

# 날짜 컬럼
date_col = df.columns[0]

# 각 자산의 종가(현재가) 추출
prices_data = {'Date': pd.to_datetime(df[date_col], errors='coerce')}

for idx, asset_name in enumerate(asset_names):
    # 종가 컬럼은 각 자산의 4번째 컬럼 (0-indexed: 0=일자, 1=시가, 2=고가, 3=저가, 4=현재가)
    col_idx = asset_indices[idx] + 4
    if col_idx < len(df.columns):
        close_col = df.columns[col_idx]
        prices_data[asset_name] = pd.to_numeric(df[close_col], errors='coerce')
        print(f"  {asset_name}: Column '{close_col}'")

prices_df = pd.DataFrame(prices_data)
prices_df = prices_df.set_index('Date')

print(f"\nPrices DataFrame shape: {prices_df.shape}")
print(f"\nFirst 10 rows:")
print(prices_df.head(10))

print(f"\nLast 10 rows:")
print(prices_df.tail(10))

print(f"\nData summary:")
print(prices_df.describe())

# CSV로 저장
output_file = "multi_asset_prices.csv"
prices_df.to_csv(output_file)
print(f"\n✓ Prices data saved to: {output_file}")

# Asset names 확인
print("\n" + "="*80)
print("Assets List:")
print("="*80)
for i, name in enumerate(asset_names, 1):
    non_null_count = prices_df[name].notna().sum()
    print(f"{i:2d}. {name:30s} - {non_null_count:,} non-null values")
