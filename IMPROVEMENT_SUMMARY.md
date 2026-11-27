# Tổng hợp cải thiện — plant-disease-diagnosis

> **Tiến độ: ⏳ ĐANG CẬP NHẬT** · cập nhật lần cuối 2026-11-27 16:20 UTC.
> Dòng này sẽ đổi thành **✅ DONE** khi tất cả hạng mục của đợt 2 hoàn tất.

File này **chỉ để đọc và ra quyết định**. Nó liệt kê từng vấn đề của repo, mức nghiêm trọng, effort, và đã được sửa ở đợt nào. Nhóm đọc file này, chọn phương án, rồi merge PR tương ứng. Nhánh `docs/improvement-summary-20261127` chỉ chứa file này và không thay đổi code.

## 1. Chọn phương án nào

| Phương án | Nhánh | PR | Gồm những gì |
| --- | --- | --- | --- |
| **Đợt 1** | [`review/repo-review-20261127`](https://github.com/thutrang147/plant-disease-diagnosis/tree/review/repo-review-20261127) | *(đang mở PR)* | Review toàn diện; sửa cách trình bày kết quả (strata, CI, Holm); đo Lab retention; LICENSE; dọn docs |
| **Đợt 2** | [`review/repo-review-round2-20261127`](https://github.com/thutrang147/plant-disease-diagnosis/tree/review/repo-review-round2-20261127) | *(đang mở PR)* | **Toàn bộ đợt 1**, cộng thêm bằng chứng mới (External v3, đo phụ thuộc vùng tổn thương, nội suy trọng số, DINOv2), tải checkpoint, tái lập, demo |

- Đợt 2 đã chứa đợt 1, nên **chỉ merge một trong hai PR**.
- Xem riêng phần thay đổi của từng đợt: [diff đợt 1](https://github.com/thutrang147/plant-disease-diagnosis/compare/main...review/repo-review-20261127) · [diff riêng đợt 2](https://github.com/thutrang147/plant-disease-diagnosis/compare/review/repo-review-20261127...review/repo-review-round2-20261127).
- Muốn chỉ lấy một vài hạng mục thì cherry-pick commit tương ứng (xem mục 4).

## 2. Thang đo

| Cột | Giá trị |
| --- | --- |
| Nghiêm trọng | 🔴 Nghiêm trọng: có thể làm sai kết luận chính · 🟠 Cao: làm yếu bằng chứng hoặc mất điểm rubric rõ ràng · 🟡 Trung bình · ⚪ Thấp |
| Ưu tiên | **P0** sửa trước khi nộp · **P1** nên sửa · **P2** làm nếu còn thời gian |
| Effort | **S** dưới 1 ngày, không cần train · **M** 1–3 ngày hoặc phải chạy lại · **L** trên 3 ngày hoặc phải thu dữ liệu mới |
| Rubric | **C1** Kiến thức CV và độ khó (20%) · **C2** Triển khai và hoàn thành (25%) · **C3** Thiết kế thực nghiệm, controls, tính đúng (25%) · **C4** Phân tích và hạn chế (15%) · **C5** Báo cáo, demo, tái lập (15%) |
| Trạng thái | ✅ Done · ⏳ Đang làm · ⬜ Chưa làm · 👤 Cần nhóm quyết định hoặc tự làm |

## 3. Bảng vấn đề, xếp theo ưu tiên

| # | Vấn đề | Nghiêm trọng | Ưu tiên | Effort | Rubric | Đợt 1 | Đợt 2 | Còn lại / ai làm |
| --- | --- | :---: | :---: | :---: | :---: | --- | --- | --- |
| 1 | Mức tăng trên External v1 (30.65% → 53.23%) gần như hoàn toàn đến từ 16 ảnh Healthy tự chụp, không phải do phân biệt bệnh tốt hơn | 🔴 | P0 | S | C3 C4 | ✅ Báo theo strata; McNemar riêng cho 46 ảnh bệnh (p = 0.83 / 0.58) | ✅ Xác nhận trên 486 ảnh độc lập (External v3): mọi model đạt 41–44%; Lab vs Mixed p = 1.0 / 0.49 | — |
| 2 | Recall "Early Blight 11/13" của ResNet Lab là do model dồn dự đoán về một lớp, không phải năng lực phân biệt | 🔴 | P0 | S | C4 | ✅ Thêm phân bố dự đoán và precision (39/62 ảnh bị dự đoán EB, precision 0.28) | ✅ Tái lập trên v3: 300/486 ảnh (62%) bị dự đoán EB; recall BS = 0.00 | — |
| 3 | RQ4: dữ liệu không đủ để nói "Field-only > Mixed"; Lab retention, lợi ích chính của Mixed, chưa được đo | 🔴 | P0 | S–M | C3 | ✅ Thêm CI và Dev tại epoch đã chọn; **đo retention**: Lab Test 0.954 (Mixed) vs 0.814 (Field-only) | ✅ So sánh pre-register trên v3: +1.7 pp, CI [−0.1, +3.6]. Nội suy trọng số cho thấy không thay được rehearsal (Field Dev 0.266 vs 0.440 ở cùng mức retention) | — |
| 4 | Không có field test độc lập được khoá trước; Field Test 47 ảnh đã bị dùng lại (CI khoảng ±14 pp) | 🔴 | P0 | L → S–M | C3 | ⬜ Chỉ ghi thành limitation | ✅ **External v3**: 486 ảnh PlantSeg, lọc trùng với 20.738 ảnh tham chiếu, protocol commit *trước* khi chạy (CI khoảng ±4.4 pp) | 👤 Chưa có ảnh Healthy để đo specificity; muốn có cần tự thu ảnh (L) |
| 5 | Nhãn nhiễu: Field Test row 2524 và cặp Bacterial Spot ↔ Septoria | 🟠 | P0 | S | C3 C4 | ✅ Ghi trong Limitations | ✅ 13/64 ảnh trùng giữa hai dataset có nhãn khác nhau (6 ca BS↔Septoria; PlantSeg gán row 2524 là Septoria); đã có sẵn danh sách để audit | 👤 Người có chuyên môn audit 13 ảnh trong `data/external_v3_plantseg/label_audit_candidates.csv` |
| 6 | Clone repo về không chạy được gì vì checkpoint chỉ có trên Drive | 🟠 | P0 | S | C2 C5 | ⬜ | ✅ `scripts/download_checkpoints.py`: 16 checkpoint, kiểm SHA-256; tải 10 checkpoint core mất khoảng 30 giây | — |
| 7 | Không có demo chạy được (app cũ chỉ có ô upload, không có inference) | 🟠 | P0 | S–M | C5 | ⬜ Chuyển scaffold vào `archive/` | ⏳ Demo Streamlit: dự đoán, Grad-CAM, so sánh các model | — |
| 8 | Báo cáo hoặc slide nằm ngoài repo (LaTeX) chưa có số liệu mới | 🟠 | P0 | M | C5 | ⬜ | ⬜ | 👤 Nhóm cập nhật, vì file không nằm trong repo |
| 9 | Kết luận dựa trên so sánh không công bằng (strawman); báo cáo chọn lọc | 🟠 | P1 | S | C3 C4 | ✅ Nêu cặp A1 vs R1; Holm cho cả nhóm kiểm định; báo cả mức tăng lớn nhất lẫn mức giảm lớn nhất | — | — |
| 10 | So sánh CNN và ViT bị confound: ResNet train khoảng 75% tham số, ViT chỉ khoảng 8% | 🟠 | P1 | M | C3 | ✅ Ghi limitation; thêm bảng recipe | ⏳ LP-FT khám phá: hai kiến trúc dùng chung recipe, so last-block với toàn bộ tham số | 👤 Bản chính thức cần GPU |
| 11 | Ngân sách train thấp: các run chọn epoch cuối trong khi Dev vẫn đang tăng | 🟡 | P1 | M | C3 | ✅ Ghi limitation | ⏳ Ablation 24 epoch | — |
| 12 | Lab split có ảnh không được nhóm theo lá, nên có thể có ảnh gần trùng giữa train và test, làm Lab accuracy bị phồng | 🟡 | P1 | S–M | C3 | ⬜ | ⏳ Audit pHash, báo Lab Test trên phần sạch | — |
| 13 | Giải thích cơ chế (Grad-CAM) chỉ định tính, không có mask | 🟡 | P1 | M | C1 C4 | ⬜ | ✅ Dùng mask chuyên gia: xoá tổn thương làm mất 38–52% dự đoán đúng; pointing 0.58–0.66 so với mức ngẫu nhiên 0.46 | Ảnh inpaint là OOD |
| 14 | Revision dataset và môi trường chưa pin | 🟡 | P1 | S | C5 | ⬜ | ✅ Pin PlantVillage `f808f27`; thêm `requirements-lock.txt` | — |
| 15 | Không có lệnh tái lập kết quả | 🟡 | P1 | S | C5 | ⬜ | ✅ `scripts/reproduce_round2.sh` + script so sánh (⏳ đang chạy kiểm chứng toàn bộ) | — |
| 16 | Báo cáo chưa giải thích vì sao chọn từng kỹ thuật CV | 🟡 | P1 | S–M | C1 | ⬜ | ⏳ `docs/technical_background.md` | — |
| 17 | README dài 300 dòng, người chấm khó tìm bằng chứng | 🟡 | P1 | S | C5 | ⬜ | ⏳ Bảng "rubric → bằng chứng" ở đầu README | — |
| 18 | License và quyền dữ liệu | 🟠 | P1 | S | C5 | ✅ `LICENSE` (code), `DATA_NOTICE.md`, `CITATION.cff` | — | 👤 Quyết định có xử lý lịch sử git (ảnh có watermark) hay không |
| 19 | Docs lẫn code scaffold không dùng | ⚪ | P1 | S | C2 | ✅ Đưa scaffold và docs cũ vào `archive/` | — | 👤 Đổi tên file NB14 nếu muốn |
| 20 | Chưa định vị so với nghiên cứu 2025–2026 | ⚪ | P2 | S | C1 | ⬜ | ✅ Thêm mục "Related work and positioning" | Rubric không chấm khả năng publication, nên giữ mục này ngắn |
| 21 | Thiếu baseline foundation model | ⚪ | P2 | S | C1 C3 | ⬜ | ✅ DINOv2-B đóng băng + linear probe: Field Test 0.665, v3 0.469 | — |
| 22 | Phân tích lỗi trên v3 theo kích thước tổn thương; calibration | ⚪ | P2 | S | C4 | ⬜ | ⏳ | — |
| 23 | NB02–14 chỉ chạy 1 seed | 🟡 | P2 | L | C3 | ⬜ | ⬜ | 👤 Cần GPU |
| 24 | Chưa có bản ViT của thí nghiệm có kiểm soát NB15 | 🟡 | P2 | M | C3 | ⬜ | ⬜ (một phần đã có qua LP-FT Field-only) | 👤 Cần GPU |
| 25 | Chưa commit prediction CSV canonical của NB11, NB13, NB14 | ⚪ | P2 | S | C5 | ⬜ | ⬜ | 👤 Cần file local của Trang |

## 4. Commit theo từng đợt (dùng khi muốn cherry-pick)

**Đợt 1**, từ `main` (`da5e1ba`):

| Commit | Nội dung |
| --- | --- |
| `cd98d6f` | REPO_REVIEW.md: review toàn diện |
| `91fb91e` | Review bổ sung: kiểm tra ảnh nhúng, Drive, license |
| `6ccb60c` | Báo theo strata, thống kê bổ sung, license, dọn docs |
| `33672f4` | Kết quả Lab retention cho RQ4 |
| `61c3cd1` | Cập nhật Results Summary |

**Đợt 2**, nối tiếp đợt 1:

| Commit | Nội dung |
| --- | --- |
| `8762743` | Khoá manifest và protocol External v3 (trước khi chạy) |
| `86e7993` | Script khai báo trước: DINOv2 probe, LP-FT |
| `cfc4752` | Kết quả External v3 |
| `f3e1a30` | Kết quả nội suy trọng số |
| `3e0d998` | Danh sách 13 ảnh cần audit nhãn |
| `bd52233` | Kết quả phụ thuộc vùng tổn thương và Grad-CAM pointing |
| `a9cd49b` | README/docs đợt 2; REPO_REVIEW §h "vì sao cải thiện so với đợt 1" |
| `fe7fdb7`, `9eaf2f5` | Baseline DINOv2 |
| `d6ca8c7` | Tải checkpoint, pin revision, lockfile, script tái lập |

Giải thích chi tiết vì sao từng hạng mục của đợt 2 tốt hơn đợt 1 nằm ở `REPO_REVIEW.md` §(h) trên nhánh đợt 2.
