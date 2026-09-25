# Vì sao chọn lô này?

Nguồn: `outputs/selection_round1.csv` (268 ảnh pool, chấm bằng model cold start) và
`outputs/selection_round1.jpg`.

## Top 5 nếu chỉ có ngân sách rà năm ảnh

Xét 50 dòng đầu CSV (điểm từ 0.959 xuống 0.820). Không frame nào trong top 50 có `empty=True`, tức
model đều có dự đoán box, nên tiêu chí phân biệt chính là điểm bất định và độ trùng cảnh.

| Thứ tự | Frame | Hạng CSV | Điểm | t (giây) | U / A / D | Lý do |
| ---: | --- | ---: | ---: | ---: | --- | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | 0.918 / 1.000 / 1.0 | Điểm cao nhất; 18 box mơ hồ (A = 1.0, bằng mức cao nhất pool). Đoạn giữa video chưa có ảnh nào được chọn. |
| 2 | frame_0369.jpg | 2 | 0.9324 | 147.6 | 0.932 / 0.889 / 1.0 | Cảnh đông (43 box dự đoán, 16 mơ hồ); đại diện đoạn cuối video. |
| 3 | frame_0331.jpg | 5 | 0.9154 | 132.4 | 0.831 / 1.000 / 1.0 | 47 box, 18 mơ hồ. **Chọn thay cho frame_0326** (hạng 4, 0.9155, t = 130.4): hai ảnh chỉ cách 2.0 giây, camera đứng yên nên gần như trùng cảnh; 0331 có nhiều box mơ hồ hơn (18 so với 15) và điểm chỉ kém 0.0001. |
| 4 | frame_0312.jpg | 7 | 0.9100 | 124.8 | 0.820 / 1.000 / 1.0 | A = 1.0 (18 box mơ hồ); cách frame_0331 7.6 giây nên là cảnh khác. |
| 5 | frame_0099.jpg | 8 | 0.9063 | 39.6 | 0.946 / 0.778 / 1.0 | U cao nhất trong top 10 (0.946); là ảnh duy nhất phủ đoạn đầu video (trước giây 70). |

Bỏ qua **frame_0372** (hạng 6, 0.9101) và **frame_0380** (hạng 3, 0.917): 0372 chỉ cách 0369 là
1.2 giây, 0380 cách 0369 là 4.4 giây. Với ngân sách năm ảnh, rà thêm một ảnh gần trùng với 0369 tốn
công ngang một ảnh mới nhưng thêm ít thông tin, nên ưu tiên trải đều theo thời gian (39.6 → 147.6 giây).

## Ba frame thuộc lô 12 ảnh model chọn và bằng chứng

- **frame_0182.jpg** (hạng 1, score 0.9591): U = 0.918, A = 1.0, 28 box dự đoán trong đó 18 box có
  0.15 ≤ conf < 0.5. Khi sửa, ảnh này có 13 box gợi ý và tôi phải **thêm 11 box** (`round1_diff.md`),
  xác nhận model bất định ở đây vì bỏ sót nhiều xe.
- **frame_0369.jpg** (hạng 2, score 0.9324): 43 box dự đoán, 16 mơ hồ. Đây là ảnh tôi thêm nhiều box
  nhất (14 gợi ý → 34 box, **thêm 20**), phần lớn là xe nhỏ ở làn trái phía xa.
- **frame_0331.jpg** (hạng 5, score 0.9154): 47 box dự đoán, 18 mơ hồ. Ảnh có nhiều box giả nhất:
  **xoá 5 box** (box trùng, box ôm mặt đường) và thêm 13.

Trên `selection_round1.jpg`, 12 ảnh được chọn trải từ giây 39.6 đến 156.8, cặp gần nhất (0326 và
0331) cách nhau 2.0 giây, đúng bằng `MIN_GAP_S`.

## Một frame điểm cao nhưng không chọn, và một frame điểm thấp vẫn nên xem

- **frame_0372.jpg** (hạng 6, 0.9101) có điểm cao hơn 6 ảnh đã được chọn nhưng **không vào lô**, vì
  chỉ cách frame_0369 (đã chọn) 1.2 giây, dưới `MIN_GAP_S = 2.0`. Tôi đồng ý: hai ảnh gần như cùng
  một dòng xe, rà cả hai gần như gấp đôi công mà mô hình học thêm rất ít.
- **frame_0195.jpg** (hạng 268, điểm thấp nhất 0.5721, 20 box, chỉ 3 box mơ hồ) vẫn nên xem một lần.
  Điểm thấp nghĩa là model *tự tin*, không phải model *đúng*: cold start chỉ có recall 0.489 trên test,
  nên nhiều xe bị bỏ sót mà không để lại box mơ hồ nào. Uncertainty sampling không nhìn thấy loại lỗi
  "bỏ sót tự tin" này.

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

Điểm cao chỉ cho biết model *đang phân vân* trên ảnh đó, không chứng minh gán nhãn ảnh đó sẽ làm AP50
tăng. Bằng chứng: sau khi fine-tune trên đúng 12 ảnh này, AP50 trên test **giảm** từ 0.771 xuống
0.452 (`rounds_table.md`). Phép chọn cũng không đo được xe bị bỏ sót hoàn toàn (không có box nên
không có độ bất định), và điểm dựa trên confidence của một model COCO chưa được hiệu chỉnh cho video
ban đêm này.
