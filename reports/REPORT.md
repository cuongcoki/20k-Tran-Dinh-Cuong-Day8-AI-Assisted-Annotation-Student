# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Trần Đình Cương

Công cụ gán nhãn đã dùng: CVAT Docker local (import/export Ultralytics YOLO Detection 1.0)

Nguồn số liệu: `reports/rounds_table.md`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`,
`outputs/selection_round1.csv`, `outputs/round1_diff.md`, `outputs/compare_round0.jpg`,
`outputs/compare_round1.jpg`. Colab chạy trên Tesla T4.

## 1. Dữ liệu và cách chia tập

Video được quay bằng camera đứng yên, lấy 2.5 ảnh/giây, nên hai ảnh liền nhau (cách 0.4 giây) gần như
giống hệt và mỗi chiếc xe nằm trong khung hình nhiều giây. Vì thế pool (268 ảnh) và test (20 ảnh, 4
đoạn quanh giây 20, 60, 100, 140) được chia theo trục thời gian, có vùng đệm 112 ảnh bị bỏ; ảnh pool
gần test nhất vẫn cách 4.4 giây (`data/DATA.md`).

Nếu chia ngẫu nhiên, cùng một chiếc xe ở cùng vị trí sẽ xuất hiện ở cả ảnh train lẫn ảnh test. Mô hình
được chấm trên những xe nó đã "thấy", nên AP50 test sẽ bị **lệch lên (quá lạc quan)** so với khả năng
thật khi gặp đoạn video mới. Đây là rò rỉ dữ liệu (data leakage).

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Tại conf 0.25, model có 197 TP, 16 FP, 206 FN trên 403 box tham chiếu (`metrics_round0.json`). Model rất
"cẩn thận": khi đã báo thì thường đúng (P = 0.925) nhưng bỏ sót hơn nửa số xe (R = 0.489).

Trên `compare_round0.jpg`, model không khớp tham chiếu ở ba nhóm:
- **Xe nhỏ ở xa, chỉ thấy đèn hậu đỏ** (góc trên-trái, gần chân trời): recall small chỉ 0.182 (66 box).
- **Xe lớn ở sát camera bị chói đèn pha** hoặc bị cắt mép ảnh: ví dụ frame_0050 và frame_0250, xe to ở
  góc dưới-trái bị tô vàng (FN). Recall large chỉ 0.561 dù xe rất rõ, cho thấy model COCO kém với xe nhìn
  từ trên xuống, bị lóa đèn ban đêm.
- **Cụm đèn hậu dày đặc**: frame_0150 (bên phải) và frame_0350 (bên trái) có box đỏ (FP) do model vẽ
  box trùng hoặc box gộp nhiều xe.

Ca cần người rà lại tham chiếu trước khi kết luận model sai: ở **frame_0250**, tham chiếu có một box nhỏ
sát **mép phải ảnh** (x ≈ 1270) chỉ gồm một vệt sáng. Có thể đó là xe bị cắt mép, cũng có thể là đèn
phản chiếu, mà guideline quy định không gán. Nhãn test do model tạo, chưa được người rà, nên một FN ở
đây chưa chắc là lỗi của model.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D` với trọng số 0.5 / 0.3 / 0.2:
- **U** (độ bất định): trung bình `1 − |2·conf − 1|` của 5 box khó nhất; cao nhất khi conf ≈ 0.5, tức
  model đang "phân vân không biết có phải xe không".
- **A** (số box mơ hồ, 0.15 ≤ conf < 0.5): chuẩn hoá theo max của pool; ảnh có nhiều chỗ phân vân thì
  một lần rà sửa được nhiều lỗi.
- **D** (đa dạng thời gian): khoảng cách tới ảnh đã gán nhãn gần nhất, tối đa 10 giây. Ở vòng 1 chưa có
  nhãn nên D = 1.0 cho mọi ảnh.
- **MIN_GAP_S = 2.0**: hai ảnh trong cùng lô phải cách nhau ít nhất 2 giây, để không tốn công rà hai
  ảnh gần như giống hệt.

Dẫn chứng (chi tiết trong `SELECTION.md`): **frame_0182** (hạng 1, 0.9591, A = 1.0), **frame_0369**
(hạng 2, 0.9324, 43 box) và **frame_0331** (hạng 5, 0.9154, 18 box mơ hồ) được chọn. **frame_0372**
(hạng 6, 0.9101) bị loại dù điểm cao, vì chỉ cách frame_0369 1.2 giây. Trong ngân sách giả định 5 ảnh,
tôi cũng bỏ frame_0326 vì trùng cảnh với frame_0331 (cách 2.0 giây), và dành chỗ cho frame_0099
(giây 39.6) để phủ đoạn đầu video.

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện mô hình. Nó chỉ đo sự phân vân của model hiện
tại, không đo xe bị bỏ sót hoàn toàn (không có box thì không có độ bất định). Vòng này là phản chứng:
fine-tune trên 12 ảnh điểm cao nhất nhưng AP50 test vẫn giảm (mục 4).

## 4. Các vòng học chủ động (active learning)

Bảng từ `rounds_table.md` (20 ảnh test, 403 box tham chiếu, bỏ 14 box cao dưới 16 px; IoU 0.5; P/R/F1
tại conf 0.25):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 307 | 0.452 | -0.319 | 1.000 | 0.127 | 0.225 | 0.000 | 0.091 | 0.585 |

**Mức sửa nhãn gợi ý vòng 1** (`round1_diff.md`): model đề xuất 169 box, sau khi sửa còn 307 box.
Giữ nguyên 157, chỉnh 4, xoá 8 (FP của model), **thêm 146** (FN của model). Accept rate 93% nghe cao,
nhưng thực tế gần **một nửa số xe trong nhãn cuối (146/307 = 48%) là do tôi tự thêm**. Lỗi chính của
nhãn AI ban đầu là bỏ sót, không phải vẽ sai.

**AP50 vòng 1** giảm 0.319 so với cold start (vòng trước cũng chính là cold start). Theo nhóm xe:
- Xe lớn: recall tăng nhẹ, 0.561 → 0.585.
- Xe trung bình: giảm mạnh, 0.547 → 0.091.
- Xe nhỏ: 0.182 → 0.000.
- Precision lên 1.000 (0 FP), nhưng chỉ còn 51 TP so với 197.

**Một ca kết quả đổi** (`compare_round1.jpg`): ở **frame_0250**, xe lớn ở góc dưới-trái là FN (vàng) với
cold start nhưng trở thành TP (xanh) ở vòng 1. Model đã học được kiểu xe to bị lóa đèn nhìn từ trên
cao, đúng với loại xe tôi đã chỉnh trong lô, và khớp với recall large tăng. Ngược lại, trên cả 4 ảnh
mẫu, model vòng 1 chỉ còn đúng **3 TP/ảnh**, bỏ gần hết xe ở giữa và ở xa, ví dụ frame_0350 có 20 FN
so với 14 FN ở cold start.

Lý do có thể kiểm tra:
1. Fine-tune từ `yolov8n.pt` với 1 lớp thay 80 lớp COCO nên **đầu phân lớp bị khởi tạo lại**. 12 ảnh ×
   50 epoch không đủ để học lại, nên confidence phần lớn dự đoán tụt dưới 0.25. AP50 (0.452) cao hơn
   nhiều so với mức recall 0.127 gợi ý, chứng tỏ vẫn còn dự đoán đúng nhưng ở conf thấp.
2. Nhãn tham chiếu test do một model tạo ra, nên cold start (cũng là model) có lợi thế khớp "phong cách"
   box. Nhãn của tôi thêm nhiều xe nhỏ và box ôm cả thân xe tối, nên lệch phong cách so với tham chiếu.

**Phân biệt ba loại bằng chứng:**
- *Quan sát độc lập* (`BLIND_SCAN.md`, khoá trước khi xem pre-label): ở frame_0099 tôi đếm 23 xe và dự
  đoán AI sẽ bỏ sót xe tối sát lề trái, xe nhỏ góc trên-trái, xe bị cắt mép phải.
- *Lỗi pre-label đã sửa*: frame_0099 có 13 box gợi ý, **nhãn cuối 23 box**, khớp đúng số xe tôi đếm
  độc lập. Tôi thêm 10 box, trong đó có xe con bật đèn ở làn giữa-trái và nhiều xe nhỏ góc trên-trái,
  đúng các vùng đã dự đoán (`REVIEW_LOG.csv`).
- *Kết quả model sau train*: là số đo trên test, tách biệt với chất lượng nhãn. Nhãn tốt hơn nhưng
  AP50 vẫn giảm, vì nguyên nhân nằm ở cách train, lượng dữ liệu và tham chiếu (xem trên).

**Ca khó theo guideline và cách tôi xử lý nhất quán:**
- *Xe buýt ở frame_0392*: box AI chỉ ôm nửa trước. Xe buýt thuộc lớp `car`, nên tôi kéo box ôm hết thân
  nhìn thấy.
- *Box gộp hai xe sát nhau ở frame_0312* (182×96 px): tôi xoá và giữ box riêng cho từng xe, theo luật
  "hai xe đứng sát nhau vẽ hai box riêng".
- *Xe chỉ thấy đèn*: ở mọi ảnh, tôi vẽ box theo thân xe đoán được quanh cụm đèn, không khoanh riêng
  chấm đèn và không tính vệt đèn trên mặt đường.

## 5. Kết luận và giới hạn

**So với cold start**, vòng 1 kém hơn rõ rệt: AP50 0.771 → 0.452, recall 0.489 → 0.127. Mức giảm 0.319
lớn hơn nhiều so với ngưỡng nhiễu khoảng 0.01 của tập test 20 ảnh, nên đây là sự suy giảm thật dưới cách
train hiện tại, không phải dao động ngẫu nhiên.

**Quyết định: dừng, không sửa nhãn vòng 2 ngay.** Notebook đã tạo lô vòng 2 (12 ảnh, giây 0.8–154.8)
nhưng chỉ có **55 box gợi ý cho 12 ảnh** (khoảng 4.6 box/ảnh, trong khi thực tế mỗi ảnh có 20–30 xe).
Model vòng 1 quá yếu nên pre-label gần như vô dụng: người gán phải vẽ lại gần hết, và điểm bất định
của một model yếu cũng khó tin cậy. Thêm 12 ảnh với cùng cách train chưa chắc cứu được, vì vấn đề nằm ở
cách fine-tune.

**Hai ca còn yếu hoặc bất định cho vòng sau:**
1. **Xe nhỏ ở xa, chỉ thấy đèn hậu** (recall small = 0.000). Ví dụ frame_0387 (hạng 3 vòng 2, A = 1.0,
   7/9 box mơ hồ). Chi phí rà cao: mỗi ảnh có nhiều xe nhỏ khó vẽ chính xác. Nguy cơ trùng cảnh: 0387
   chỉ cách frame_0392 (đã gán) 2 giây, D = 0.2, nên nên cân nhắc bỏ.
2. **Cụm xe dày ở làn giữa, cỡ trung bình** (recall medium 0.547 → 0.091). Ví dụ frame_0124 (hạng 1 vòng
   2, score 0.771, 5/7 box mơ hồ, giây 49.6, D = 0.68), một đoạn thời gian chưa có nhãn. Nên chọn ảnh
   này. Ngược lại, **không chọn frame_0368** (hạng 19 vòng 2): chỉ cách frame_0369 đã gán 0.4 giây
   (D = 0.04), gần như trùng ảnh.

**Giới hạn ảnh hưởng tới kết luận:**
- Test chỉ 20 ảnh từ 4 đoạn thời gian, nên chưa đại diện mọi mật độ xe. Có thể so thứ tự lớn/nhỏ giữa các
  vòng, nhưng không nên tin chênh lệch nhỏ.
- 14 box cao dưới 16 px bị bỏ qua, nên số đo không phản ánh khả năng thấy xe ở chân trời.
- Nhãn tham chiếu do model tạo, chưa được người rà. AP50 đo *mức khớp với tham chiếu*, không phải độ
  đúng tuyệt đối. Tôi thêm 146 xe mà AI bỏ sót; nếu tham chiếu cũng bỏ sót các xe tương tự, model học
  đúng những xe đó vẫn không được cộng điểm.

**Việc kiểm tra trước khi train thêm (vì AP50 giảm):**
1. Hạ ngưỡng conf khi dự đoán test (0.05–0.1) xem recall có hồi lại không, để xác nhận vấn đề là
   confidence chưa hiệu chỉnh hay model không thấy xe.
2. Thử fine-tune từ trọng số cold start mà vẫn giữ kiến thức COCO (freeze backbone, ít epoch hơn, hoặc
   giữ lớp COCO rồi gộp), để tránh khởi tạo lại đầu ra với chỉ 12 ảnh.
3. Rà lại nhãn của mình: tìm box trùng hoặc lệch và kiểm tra tính nhất quán giữa các ảnh (ví dụ cụm box
   cùng cỡ 46×37 px ở frame_0369). Không sửa nhãn test hay file số đo.
