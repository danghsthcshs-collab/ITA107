import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# --- CẤU HÌNH TÊN FILE ---
# Thay 'ten_file_cua_ban.csv' bằng tên file thực tế của bạn
file_path = 'ten_file_cua_ban.csv' 

# Kiểm tra file có tồn tại không trước khi đọc
if not os.path.exists(file_path):
    print(f"LỖI: Không tìm thấy file '{file_path}'.")
    print(f"Hãy đặt file .csv vào cùng thư mục với script này: {os.getcwd()}")
else:
    df = pd.read_csv(file_path)
    print("Đã tải dữ liệu thành công!")

    # ==========================================
    # BÀI 1: KHÁM PHÁ DỮ LIỆU
    # ==========================================
    print("\n--- BÀI 1: KHÁM PHÁ ---")
    print(df.head(10))
    print(f"\nKích thước: {df.shape}")
    print("\nThông tin kiểu dữ liệu:")
    print(df.info())
    print("\nThống kê mô tả:")
    print(df.describe())
    
    # Histogram
    df.hist(figsize=(10, 6), bins=20)
    plt.suptitle("Histogram")
    plt.show()

    # ==========================================
    # BÀI 2: LÀM SẠCH DỮ LIỆU
    # ==========================================
    print("\n--- BÀI 2: LÀM SẠCH ---")
    
    # Xử lý giá trị thiếu
    df = df.fillna(df.median(numeric_only=True))
    
    # Xóa trùng lặp
    df.drop_duplicates(inplace=True)
    
    # Boxplot trước chuẩn hóa
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df.select_dtypes(include=[np.number]))
    plt.title("Boxplot trước chuẩn hóa")
    plt.show()
    
    # Chuẩn hóa (Normalization)
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Boxplot sau chuẩn hóa
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df[num_cols])
    plt.title("Boxplot sau chuẩn hóa")
    plt.show()
    
    print("\nHoàn thành xử lý Bài 1 và Bài 2!")
                
                