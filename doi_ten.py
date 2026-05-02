import os
import re
import pandas as pd

# --- CẤU HÌNH ĐƯỜNG DẪN ---
csv_path = 'danh_sach_van_ban_dut.csv'
dir_path = 'data/cleaned'

# Tên cột trong file CSV (bạn hãy sửa lại nếu tên cột thực tế khác)
COL_ID = 'ID'
COL_SO_HIEU = 'So_Hieu' # Có thể là 'Số/Ký hiệu' hoặc 'Ký hiệu' tùy file của bạn

# --- 1. ĐỌC VÀ XỬ LÝ DỮ LIỆU CSV ---
df = pd.read_csv(csv_path)

# Xóa dấu '/' trong cột Số hiệu để khớp với định dạng trên tên file
df['clean_so_hieu'] = df[COL_SO_HIEU].astype(str).str.replace('/', '', regex=False)

# Tạo dictionary map từ 'Số hiệu (đã xóa gạch chéo)' -> 'ID'
so_hieu_to_id = dict(zip(df['clean_so_hieu'], df[COL_ID]))

# --- 2. QUÉT VÀ ĐỔI TÊN FILE ---
# Regex giải thích:
# data1_ : Cố định ở đầu
# (\d+)  : Nhóm 1 - Lấy số thứ tự cũ (vd: 0447)
# (.*?)  : Nhóm 2 - Lấy đoạn số hiệu văn bản (vd: 842025QH15)
# _AI_Corrected\.txt : Cố định ở cuối
pattern = re.compile(r'^data1_(\d+)_(.*?)_AI_Corrected\.txt$')

# Đếm số lượng file đã đổi tên thành công
success_count = 0

for filename in os.listdir(dir_path):
    match = pattern.match(filename)
    
    if match:
        old_num = match.group(1)
        so_hieu_file = match.group(2)
        
        # Tìm xem số hiệu trên file có nằm trong file CSV không
        if so_hieu_file in so_hieu_to_id:
            new_id = str(so_hieu_to_id[so_hieu_file])
            
            # Pad thêm số 0 ở đầu cho đủ 4 chữ số (vd: ID 42 -> 0042)
            # Bạn có thể bỏ .zfill(4) nếu chỉ muốn dùng số gốc
            new_id_padded = new_id.zfill(4)
            
            # Tạo tên file mới
            new_filename = f"data1_{new_id_padded}_{so_hieu_file}_AI_Corrected.txt"
            
            old_filepath = os.path.join(dir_path, filename)
            new_filepath = os.path.join(dir_path, new_filename)
            
            # Tiến hành đổi tên
            if old_filepath != new_filepath:
                os.rename(old_filepath, new_filepath)
                print(f"[OK] Đã đổi: {filename} -> {new_filename}")
                success_count += 1
        else:
            print(f"[WARN] Không tìm thấy '{so_hieu_file}' trong CSV. Bỏ qua file: {filename}")
    else:
        # Bỏ qua các file không khớp định dạng chuẩn (ví dụ: Don_chuyenCTDT_...)
        pass

print(f"\nĐã xử lý xong. Tổng số file được đổi tên: {success_count}")