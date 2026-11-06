# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603980
- **Họ và tên:** Hoàng Thế Duyệt
- **SEED** (in ở ô đầu notebook): 7325

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | ---: | ---: | ---: | ---: |
| 1. BoW + LogReg | 0.890 | 0.889 | 0.06 | 0 |
| 2. BoW + MLP | 0.955 | 0.947 | 0.26 | 0 |
| 3a. Encoder + kNN | 0.888 | 0.974 | 68.1 | 0 |
| 3b. Encoder + LogReg | 0.956 | 1.000 | 17.0 | 0 |
| 3c. Encoder + MLP head | 0.956 | 1.000 | 76.2 | 0 |
| 4a. LLM nhỏ zero-shot | — | 0.930 | 877 | 0.026 |
| 4c. LLM nhỏ few-shot | — | 0.952 | 945 | 0.061 |

LLM dùng Gemini (gemini-3.5-flash-lite). Không có hàng 4b vì model lớn bị giới hạn quota gói miễn phí nên mình dừng.

## 3. Dự đoán và thực tế

- **Bỏ ReLU (2.B):** đoán F1 giảm về gần LogReg, thực tế vẫn 0.955. Lệch vì MLP hơn LogReg ở Phần 1 là nhờ train bằng Adam + mini-batch chứ không phải nhờ lớp ẩn.
- **50 case khó:** đoán LLM chỉ đúng khoảng 80% và bị prompt injection lừa, thực tế đúng 50/50. Có thể model mới đã được huấn luyện chống kiểu này, nhưng 6 case là quá ít để kết luận.
- **lr = 50 (1.B):** đoán loss dao động mạnh, thực tế chỉ vọt lên vài epoch đầu rồi về 0.004. Lệch vì loss của logistic regression là hàm lồi nên vẫn hội tụ.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 từ 0.889 xuống 0.333. Nhóm khó nhất với nó là spam giọng tự nhiên (chỉ đúng 1/10) và né từ khoá (1/6).

- Spam giọng tự nhiên, ví dụ #14 "Hi, I'm working from home and making about £300 a day...": không có từ free/call/txt mà lại toàn từ bình thường như I, from, home nên BoW kéo về ham.
- Né từ khoá, ví dụ #29 "C0ngr4tulat10ns! Y0u h4ve w0n...": các từ bị đổi chữ thành số nên không có trong từ điển 2.000 từ, BoW gần như không thấy gì.

Embedding làm tốt hơn ở nhóm né từ khoá (83–100%) nhưng vẫn yếu ở spam giọng tự nhiên vì về nghĩa những tin này giống tin người quen.

## 5. Thử thách cá nhân

Đề #6: cho LLM trả JSON có label và reason, so với bản trả 1 từ trên 10 tin SUB. Cả hai đều đúng 10/10, nhưng bản JSON tốn token ra khoảng 204 so với 13, chi phí gấp 2,9 lần và chậm hơn khoảng 126 ms/tin. Không đáng dùng cho mọi tin, chỉ nên dùng khi cần giải thích lý do chặn.

## 6. Phương án cho sản phẩm thật

Chọn 3b (Encoder + LogReg): F1 test 0.956, p95 17 ms, đạt yêu cầu dưới 50 ms và chạy được hơn 1.300 tin/s, dư cho 116 tin/s. Trên 50 case khó nó cũng tốt hơn BoW nhiều (0.67 so với 0.33). Đánh đổi là cần GPU và chậm hơn BoW. LLM thì không đạt độ trễ (gần 1 giây/tin) và tốn khoảng 7.700 đô/tháng, nên chỉ dùng cho tin lưng chừng.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | Nhận gì | Mình làm gì |
| --- | --- | --- | --- |
| 0 | Data leakage, đếm tin trùng | Giải thích | Tự viết câu trả lời |
| 1–3 | Code còn yếu, gấp thời gian nên nhờ viết code các TODO | Code + giải thích | Đọc hiểu, tự tính lại ví dụ, chạy ô kiểm tra |
| 4 | Code gọi API, sửa lỗi 429/400 của Gemini | Code, thêm nghỉ giữa các lần gọi | Đọc trước khi chạy |
| Câu hỏi | Gợi ý trả lời theo số liệu của mình | Bản nháp | Sửa lại, so số với output |
| Thử thách #6 | Nhờ viết code (đề không cho dùng AI, mình ghi rõ) | Code | Đọc hiểu từng hàm |

Hôm nay mình dựa vào AI nhiều hơn mức lab cho phép vì còn yếu NumPy/PyTorch, buổi sau sẽ tự viết trước. Điều mình hiểu ra: phải đổi từng thứ một mới biết cái gì tạo ra khác biệt, và điểm cao trên test thường chưa chắc mô hình đã hiểu, phải thử với case khó mới thấy.
