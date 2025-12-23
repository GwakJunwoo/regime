"""
CSV 파일 구조 분석 스크립트
"""
import pandas as pd
import chardet

# 파일 인코딩 감지
file_path = "가격 데이터.csv"

with open(file_path, 'rb') as f:
    result = chardet.detect(f.read(100000))
    print(f"Detected encoding: {result}")

# 다양한 인코딩으로 시도
encodings = ['cp949', 'euc-kr', 'utf-8', 'utf-16', 'latin1']

for enc in encodings:
    try:
        print(f"\n{'='*60}")
        print(f"Trying encoding: {enc}")
        print('='*60)
        
        df = pd.read_csv(file_path, encoding=enc, nrows=10)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")
        
        print(f"\nFirst 3 rows:")
        print(df.head(3))
        
        print(f"\n✓ Successfully loaded with {enc}")
        
        # 성공하면 전체 데이터 로드
        df_full = pd.read_csv(file_path, encoding=enc)
        print(f"\nFull data shape: {df_full.shape}")
        
        # 날짜 컬럼 찾기
        date_cols = [col for col in df_full.columns if '날짜' in str(col).lower() or 'date' in str(col).lower()]
        print(f"Date columns: {date_cols}")
        
        # 종가 컬럼 찾기
        close_cols = [col for col in df_full.columns if '종가' in str(col) or 'close' in str(col).lower()]
        print(f"Close price columns: {close_cols}")
        
        break
        
    except Exception as e:
        print(f"✗ Failed with {enc}: {str(e)[:100]}")
        continue
