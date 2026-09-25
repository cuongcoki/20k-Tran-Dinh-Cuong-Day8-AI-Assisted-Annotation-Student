# Quét độc lập trước khi xem pre-label

Frame: frame_0099.jpg

Số xe nhìn thấy bằng mắt: 23

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe: 

Xe ở mép phải ảnh, chỉ xuất hiện một phần và bị cắt bởi khung hình.
Xe ở phía trái gần mép đường, tối và khó phân biệt với nền, chủ yếu thấy ánh đèn pha.
xe ở phía xa 
Góc trên bên trái ảnh: Có một số xe xuất hiện ở khu vực này nhưng bị tối, kích thước nhỏ và một phần bị khuất bởi vùng tối, nên AI dễ bỏ sót hoặc vẽ sai vị trí/bounding box.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
