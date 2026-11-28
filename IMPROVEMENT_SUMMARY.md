# Tổng hợp cải thiện — plant-disease-diagnosis

> **Tiến độ: ⏳ ĐANG CẬP NHẬT** · cập nhật lần cuối 2026-11-28 00:05 UTC (07:05 giờ VN).
> Dòng này đổi thành **✅ DONE** khi đợt 2 đóng băng (dừng chạy thí nghiệm 03:30 UTC, push cuối 04:15 UTC, tức 11:15 giờ VN).

File này **chỉ để đọc và ra quyết định**. Nó liệt kê:
- từng vấn đề của repo, mức nghiêm trọng, effort, và đã được sửa ở đợt nào;
- điểm chấm ước lượng theo rubric từ một hội đồng 5 reviewer độc lập;
- những việc chỉ nhóm mới làm được.

Nhánh `docs/improvement-summary-20261127` chỉ chứa file này và không thay đổi code.

## 1. Chọn phương án nào

| Phương án | Nhánh | PR | Gồm những gì |
| --- | --- | --- | --- |
| **Đợt 1** | [`review/repo-review-20261127`](https://github.com/thutrang147/plant-disease-diagnosis/tree/review/repo-review-20261127) | [#6](https://github.com/thutrang147/plant-disease-diagnosis/pull/6) | Review toàn diện; sửa cách trình bày (strata, CI, Holm); đo Lab retention; LICENSE; dọn docs. Đã được sửa thêm theo hội đồng reviewer (commit `c90d5f3`). |
| **Đợt 2** (khuyến nghị) | [`review/repo-review-round2-20261127`](https://github.com/thutrang147/plant-disease-diagnosis/tree/review/repo-review-round2-20261127) | [#7](https://github.com/thutrang147/plant-disease-diagnosis/pull/7) | **Toàn bộ đợt 1**, cộng thêm: test độc lập pre-register (External v3, 486 ảnh); các phép kiểm có đối chứng; tải checkpoint có kiểm SHA; script tái lập; demo chạy thật; tài liệu kỹ thuật; sửa theo hội đồng reviewer và Leader. |

- **Chỉ merge một trong hai PR.** Đợt 2 đã chứa toàn bộ đợt 1, kể cả các bản sửa của đợt 1.
- Xem riêng phần thay đổi: [diff đợt 1](https://github.com/thutrang147/plant-disease-diagnosis/compare/main...review/repo-review-20261127) · [diff riêng đợt 2](https://github.com/thutrang147/plant-disease-diagnosis/compare/review/repo-review-20261127...review/repo-review-round2-20261127).
- Muốn lấy một phần thì cherry-pick commit tương ứng ở mục 6.

## 2. Thang đo

| Cột | Giá trị |
| --- | --- |
| Nghiêm trọng | 🔴 Có thể làm sai kết luận chính · 🟠 Làm yếu bằng chứng, hoặc mất điểm rubric rõ ràng · 🟡 Trung bình · ⚪ Thấp |
| Ưu tiên | **P0** sửa trước khi nộp · **P1** nên sửa · **P2** làm nếu còn thời gian |
| Effort | **S** dưới 1 ngày, không cần train · **M** 1–3 ngày, hoặc phải chạy lại · **L** trên 3 ngày, hoặc phải thu dữ liệu mới |
| Rubric | **C1** Kiến thức CV (20%) · **C2** Triển khai (25%) · **C3** Thiết kế thực nghiệm (25%) · **C4** Phân tích và hạn chế (15%) · **C5** Báo cáo, demo, tái lập (15%) |
| Trạng thái | ✅ Done · ⏳ Đang làm · ⬜ Chưa làm · 👤 Cần nhóm quyết định hoặc tự làm |

## 3. Điểm rubric ước lượng (hội đồng 5 reviewer)

Mỗi reviewer là một agent độc lập, chỉ được đọc repo, phụ trách đúng một tiêu chí, và mọi nhận định phải kèm bằng chứng (file, dòng hoặc con số). Họ chấm nhánh đợt 2 tại commit `ab02c7e`, **trước** các bản sửa ở mục 4.

| Tiêu chí | Điểm | Lý do chính bị trừ (trước khi sửa) |
| --- | ---: | --- |
| C1 Kiến thức CV (20%) | 7.0 | Phép occlusion chưa có đối chứng; không có attribution cho ViT; câu "representation > recipe" chưa đủ căn cứ |
| C2 Triển khai (25%) | 8.0 | Script tái lập gãy ở bước pytest; code train chỉ có trong notebook; vài lỗi nhỏ |
| C3 Thiết kế thực nghiệm (25%) | 7.5 | Kết quả null bị viết như "đã xác nhận"; chọn metric khác nhau ở từng phần; độc lập và nhãn của v3 chưa chứng minh |
| C4 Phân tích (15%) | 7.5 | Đổi metric theo từng claim; khoảng 8/60 con số lệch; tài liệu tự mâu thuẫn |
| C5 Báo cáo, demo, tái lập (15%) | 7.0 | Demo ghi sai "mọi model 0.97–0.99 trên lab"; tài liệu phụ lỗi thời; không có hình |
| **Trung bình có trọng số** | **≈ 7.45/10** | |

Các reviewer ước tính sửa các lỗi effort S sẽ đưa C4 lên khoảng 8.5–9 và C5 lên khoảng 8. Hầu hết các lỗi đó đã được sửa (mục 4). Phần chưa sửa được là những việc cần người hoặc GPU (mục 5).

## 4. Bảng vấn đề, xếp theo ưu tiên

| # | Vấn đề | Nghiêm trọng | Ưu tiên | Effort | Rubric | Đợt 1 | Đợt 2 | Còn lại / ai làm |
| --- | --- | :---: | :---: | :---: | :---: | --- | --- | --- |
| 1 | Mức tăng trên External v1 gần như hoàn toàn đến từ 16 ảnh Healthy tự chụp | 🔴 | P0 | S | C3 C4 | ✅ Báo theo strata; McNemar riêng cho ảnh bệnh | ✅ Kiểm lại trên 486 ảnh v3: CI của accuracy loại trừ mức tăng trên khoảng 5 pp | — |
| 2 | **Headline nói "phân biệt bệnh không cải thiện, đã xác nhận"**, trong khi macro-F1 bệnh (metric phụ đã pre-register) có tăng *(hội đồng phát hiện)* | 🔴 | P0 | S | C3 C4 | ✅ Bỏ các chữ "confirmed" và "indistinguishable" | ✅ Viết theo giới hạn CI: accuracy ±5 pp; macro-F1 +0.086 [0.039; 0.133] và +0.065 [0.029; 0.102] (file `external_v3_secondary_metrics/`) | — |
| 3 | Recall "Early Blight 11/13" là do model dồn dự đoán về một lớp | 🔴 | P0 | S | C4 | ✅ Báo phân bố dự đoán; ghi rõ 39/62 (chuẩn) và 43/62 (loader cũ) | ✅ Tái lập trên v3: 300/486 ảnh bị dự đoán EB | — |
| 4 | RQ4: kết quả null bị viết như tương đương; Lab retention chưa được đo | 🔴 | P0 | S–M | C3 | ✅ Đo retention (0.954 vs 0.814); bỏ câu "tương đương" | ✅ P3 trên v3 (8 epoch): +1.7 pp [−0.1; +3.6], p theo seed 0.12. ✅ Chạy lại 24 epoch, 3 seed (E2): Mixed vẫn giữ Lab tốt hơn +0.11 đến +0.13 ở mọi seed (p = 0.001), và cao hơn trên v3 +2.4 pp [+0.7; +4.1] (p theo seed 0.06; khám phá, v3 đã dùng lại). Train lâu hơn giúp cả hai nhánh nhiều hơn (+4.6 đến +5.3 pp) | — |
| 5 | Không có field test độc lập được khoá trước | 🔴 | P0 | L → S–M | C3 | ⬜ | ✅ External v3: 486 ảnh, lọc trùng, protocol commit trước khi chạy; có sổ ghi mọi lần đánh giá (`docs/evaluation_ledger.md`) | 👤 Chưa có ảnh Healthy nên chưa đo được specificity (L) |
| 6 | Nhãn nhiễu: row 2524 và **row 2532** của Field Test đều đang bị tranh chấp; cặp BS ↔ Septoria | 🟠 | P0 | S | C3 C4 | ✅ Ghi trong Limitations | ✅ 13/64 ảnh trùng có nhãn khác nhau (6 cặp BS ↔ Septoria); danh sách audit có sẵn. ✅ Gộp BS + Septoria: accuracy +8 đến +15 pp, P1 thành +4.5 pp [−0.8; +9.9]; đo được bao nhiêu lỗi nằm trên ranh giới này, nhưng không tách được nhiễu nhãn với lỗi model | 👤 Người có chuyên môn audit 13 ảnh (`label_audit_candidates.csv`) |
| 7 | Clone repo về không chạy được gì | 🟠 | P0 | S | C2 C5 | ⬜ | ✅ Tải checkpoint có kiểm SHA; script tái lập gọi unittest; script so sánh kiểm từng ô | 👤 Sao lưu checkpoint sang vị trí thứ hai (hiện chỉ có trên một Drive cá nhân) |
| 8 | Không có demo chạy được; demo ghi sai số liệu lab *(hội đồng phát hiện)* | 🟠 | P0 | S | C5 | ⬜ | ✅ Demo hiện số đo thật của từng model (đọc từ CSV), ảnh ViT đúng vùng model thấy | 👤 Quay video demo dự phòng; tải sẵn dataset trước khi demo trực tiếp |
| 9 | Báo cáo hoặc slide ngoài repo (LaTeX) chưa có số liệu mới | 🟠 | P0 | M | C5 | ⬜ | ⬜ | 👤 Nhóm cập nhật theo README đợt 2 |
| 10 | "Model dựa vào vùng tổn thương" chưa có đối chứng; Grad-CAM bị diễn giải quá *(hội đồng phát hiện)* | 🟠 | P1 | S | C1 C4 | ⬜ | ✅ Đối chứng xoá vùng ngẫu nhiên cùng diện tích: mức giảm thêm 0.12–0.15, CI loại trừ 0, nên **model có dựa vào tổn thương**. Xoá vùng đối chứng chỉ làm accuracy đổi −1.6 đến +1.9 pp (378 ảnh; 108 ảnh có vết bệnh quá lớn bị loại). Grad-CAM pointing (0.58–0.66) **trên mức ngẫu nhiên (0.46) nhưng không hơn baseline tâm ảnh (0.74)** | Grad-CAM cho ViT (cần thêm thời gian) |
| 11 | So sánh CNN vs ViT bị confound (tỉ lệ tham số 75% vs 8%) | 🟠 | P1 | M | C3 | ✅ Ghi limitation | ✅ Train lại khám phá: thứ hạng **phụ thuộc cả độ sâu fine-tune lẫn ngân sách train** | 👤 Bản chính thức nhiều seed cần GPU |
| 12 | Ngân sách train thấp (chọn epoch cuối khi Dev còn tăng) | 🟡 | P1 | M | C3 | ✅ Ghi limitation | ✅ Ablation 24 epoch: Field Dev của ResNet vẫn tăng (epoch được chọn 23/24), ViT plateau quanh epoch 12–18. ✅ NB15 chạy lại 24 epoch × 3 seed, cả hai nhánh; tái lập Lab Test của checkpoint niêm phong trong 0.001 (4/6 khớp từng bit) | 👤 NB02–14 vẫn ở 8 epoch; bản chính thức cần GPU |
| 13 | Lab split có thể bị rò rỉ | 🟡 | P1 | S–M | C3 | ⬜ | ✅ Chỉ 3/1.341 và 2/1.692 ảnh gần trùng; Lab Test không đổi | Không bắt được các ảnh khác nhau của cùng một lá |
| 14 | So sánh nội suy trọng số và DINOv2 bị trình bày quá *(hội đồng phát hiện)* | 🟡 | P1 | S | C3 C4 | ⬜ | ✅ Nội suy: kết luận phụ thuộc metric (accuracy v3 0.447 vs 0.434). DINOv2: confound dung lượng model và dữ liệu pretrain; đã bỏ câu "representation > recipe" | — |
| 15 | Số liệu lệch: Holm 10 phép (không phải 8); 4 hàng Field Test bị loại (không phải 3); 77+17 ảnh trùng; typo CI; số test | 🟡 | P1 | S | C4 | ✅ Holm, typo CI | ✅ Đã sửa hết | — |
| 16 | Tài liệu phụ tự mâu thuẫn (methods, results summary, DATA_NOTICE, workflow) | 🟡 | P1 | S | C5 | ⬜ | ✅ Đã đồng bộ | — |
| 17 | Revision dataset và môi trường chưa pin | 🟡 | P1 | S | C5 | ⬜ | ✅ Pin PlantVillage `f808f27`; lockfile (chỉ cho Linux x86_64) | Lockfile riêng cho CPU (chưa làm) |
| 18 | Báo cáo chưa giải thích vì sao chọn từng kỹ thuật CV | 🟡 | P1 | S–M | C1 | ⬜ | ✅ `docs/technical_background.md`, đã sửa theo hội đồng | — |
| 19 | Người chấm khó tìm bằng chứng; README không có hình | 🟡 | P1 | S | C5 | ⬜ | ✅ Quick start; bảng "Where to find the evidence"; thêm 2 hình | — |
| 20 | Calibration và phân tích lỗi trên v3 | ⚪ | P2 | S | C4 | ⬜ | ✅ Model Lab tự tin 0.90 trong khi accuracy 0.43 (ECE 0.475); model đã adapt có ECE 0.16–0.29 | — |
| 21 | Baseline foundation model | ⚪ | P2 | S | C1 C3 | ⬜ | ✅ DINOv2 probe (khám phá), có ghi rõ các confound | — |
| 22 | License và lịch sử git có ảnh watermark | 🟠 | P1 | S | C5 | ✅ LICENSE, DATA_NOTICE, CITATION | ✅ DATA_NOTICE thêm PlantSeg | 👤 Quyết định có xử lý lịch sử git hay không |
| 23 | NB02–14 chỉ chạy 1 seed; chưa có bản ViT có kiểm soát của NB15 | 🟡 | P2 | L | C3 | ⬜ | ⬜ | 👤 Cần GPU |
| 24 | Đạt macro-F1 "tốt" (≥ 0.70) trên field | — | — | L | — | — | **Cố ý không làm**, Leader quyết định. Lý do: (1) mốc 0.70 trên 47 ảnh Field Test có Wilson 95% CI khoảng [0.56; 0.81], nên "đạt 0.70" không phân biệt được với 0.60 về mặt thống kê; (2) muốn tiến gần mốc đó phải tune nhiều lần trên chính các test set đã dùng lại, làm hỏng tính trung thực của mọi kết luận; (3) fine-tune DINOv2 cần GPU, trong khi môi trường này chỉ có CPU và hạn là sáng nay. DINOv2 chỉ được chạy dạng probe đóng băng (khám phá) | Hướng làm tiếp: fine-tune DINOv2 trên GPU, crop lá (bài PlantDoc đạt 0.70 accuracy trên ảnh đã crop), thêm dữ liệu field, AdaBN rồi fine-tune |
| 25 | Chưa commit prediction CSV chuẩn của NB11/13/14 | ⚪ | P2 | S | C5 | ⬜ | ⬜ | 👤 Cần file local của Trang |
| 26 | Chưa kiểm phần khoảng cách lab → field có phải do thống kê BatchNorm không | 🟡 | P1 | S | C1 C3 | ⬜ | ✅ AdaBN (khám phá): chỉ ước lượng lại BN trên 370 ảnh field không nhãn, model Lab tăng Field Dev macro-F1 0.235 → 0.391 và v3 accuracy 0.426 → 0.459; model đã adapt thì giảm. Lab Test tụt 0.974 → 0.673 (giá phải trả) | Chưa thử "AdaBN rồi mới fine-tune" |
| 27 | Test set bị dùng lại nhiều lần | 🟠 | P1 | S | C3 | ⬜ | ✅ Sổ ghi đánh giá: 104 lần đánh giá model trên v3 sau lần chạy confirmatory, đều dán nhãn khám phá | — |

## 5. Việc chỉ nhóm làm được (checklist trước khi nộp)

- [ ] **Chọn PR để merge:** #7 (đợt 2, khuyến nghị, đã chứa đợt 1) hoặc #6 (chỉ đợt 1). Merge một PR rồi đóng PR còn lại.
- [ ] **Audit nhãn:** người có chuyên môn xem 13 ảnh bất đồng nhãn (`data/external_v3_plantseg/label_audit_candidates.csv`), ưu tiên row 2524 và row 2532 của Field Test và 6 cặp BS ↔ Septoria.
- [ ] **Báo cáo và slide:** cập nhật LaTeX và slide theo README đợt 2, đặc biệt là abstract, RQ2 (v3, AdaBN, ranh giới BS/Septoria) và RQ4 (kết quả 24 epoch).
- [ ] **Video demo:** quay video dự phòng; chạy demo một lần trước buổi bảo vệ để tải sẵn dataset và checkpoint.
- [ ] **Sao lưu checkpoint** sang vị trí thứ hai (hiện chỉ có trên một Drive cá nhân); quyết định cách xử lý lịch sử git có ảnh watermark.
- [ ] (Nếu có GPU) Chạy thêm seed cho các run 24 epoch; làm bản ViT có kiểm soát của NB15; thử AdaBN rồi mới fine-tune.

## 6. Câu hỏi examiner có thể hỏi, kèm câu trả lời trung thực (gom từ 5 reviewer)

1. **"Accuracy không đổi mà macro-F1 lại tăng?"** Accuracy (metric chính) không tăng quá khoảng 5 pp. Macro-F1 (metric phụ) tăng 0.07–0.09. Lý do: baseline dồn dự đoán về Early Blight, lớp lớn nhất. Adaptation làm thay đổi *cách* model sai nhiều hơn là *số lần* sai.
2. **"Mixed và Field-only có tương đương không?"** Không chứng minh được tương đương. Ở 8 epoch, CI trên v3 là [−0.1; +3.6] pp: loại trừ trường hợp Mixed thua và trường hợp Mixed thắng quá 3.6 pp. Ở 24 epoch (khám phá, 3 seed), Mixed cao hơn +2.4 pp [+0.7; +4.1], nhưng p theo seed là 0.06 và v3 đã bị dùng lại nhiều lần. Lợi ích chắc chắn của Mixed là giữ được kết quả trên Lab (+0.11 đến +0.14 macro-F1 ở mọi seed, cả 8 lẫn 24 epoch). Train lâu hơn giúp field accuracy nhiều hơn là thêm ảnh Lab.
3. **"Làm sao biết protocol đã khoá trước khi chạy?"** Commit `8762743` (protocol) có trước `cfc4752` (kết quả), và SHA-256 khớp với `run_record.json`. Đây là bằng chứng từ lịch sử git chứ không phải xác nhận của bên thứ ba. Mọi phân tích sau đó đều là khám phá (`docs/evaluation_ledger.md`).
4. **"External v3 có thật sự độc lập không?"** Ảnh đã được lọc với 20.738 ảnh tham chiếu và 94 ảnh bị loại. Bộ lọc pHash không bắt được ảnh bị crop hoặc lật, nên đây là cận dưới của mức trùng. Nhãn PlantSeg chưa được audit (13/64 ảnh bất đồng).
5. **"Grad-CAM có cho thấy model nhìn vào vết bệnh không?"** Không đáng tin. Model *có* dựa vào vết bệnh: xoá vết bệnh làm accuracy giảm thêm 0.12–0.15 so với xoá một vùng đối chứng cùng diện tích. Đỉnh Grad-CAM trúng vết bệnh nhiều hơn mức ngẫu nhiên (0.58–0.66 so với 0.46), nhưng ít hơn baseline "luôn chỉ vào tâm ảnh" (0.74). Kết luận này chỉ áp dụng cho Grad-CAM của ResNet18; chưa thử phương pháp attribution khác.
6. **"CNN hay ViT tốt hơn?"** Không kết luận được. Thứ hạng đổi theo độ sâu fine-tune và ngân sách train; các run 8 epoch đều là model chưa hội tụ.
7. **"Clone về thì tái lập được gì?"** Tái lập được các phân tích inference (v3, retention, nội suy, lesion, control) bằng `reproduce_round2.sh`, mất khoảng 1.5 giờ CPU và tải khoảng 4 GB. Phần train của NB02–15 cần GPU trên Colab.
8. **"Vì sao không đạt 0.70?"** Mốc đó trên 47 ảnh có CI rộng khoảng ±13 pp, và muốn đạt thì phải tune trên chính test set. Nhóm ưu tiên kết luận đáng tin hơn là một con số đẹp (xem hàng 24).
9. **"Khoảng cách lab → field đến từ đâu?"** Một phần là do thống kê BatchNorm: ước lượng lại BN trên ảnh field không nhãn (AdaBN) đã nâng Field Dev macro-F1 của model Lab từ 0.235 lên 0.391. Phần còn lại cần nhãn field. Đây là kết quả khám phá, chỉ áp dụng cho ResNet18.

## 7. Commit theo từng đợt

**Đợt 1** (từ `main` `da5e1ba`):

| Commit | Nội dung |
| --- | --- |
| `cd98d6f` | REPO_REVIEW.md |
| `91fb91e` | Review bổ sung |
| `6ccb60c` | Strata, thống kê bổ sung, license, dọn docs |
| `33672f4` | Lab retention cho RQ4 |
| `61c3cd1` | Results Summary |
| `c90d5f3` | Sửa theo hội đồng reviewer: bỏ câu "tương đương", Holm 10 phép, 39/62 và 43/62, thêm limitations |
| `4d138dd` | Banner trỏ sang đợt 2 và PR tổng hợp |

**Đợt 2**, nối tiếp đợt 1:

| Commit | Nội dung |
| --- | --- |
| `8762743` → `cfc4752` | Khoá protocol External v3, rồi chạy một lần |
| `f3e1a30` | Nội suy trọng số |
| `bd52233` | Đo phụ thuộc vùng tổn thương |
| `3e0d998` | Danh sách 13 ảnh cần audit nhãn |
| `fe7fdb7`, `9eaf2f5` | DINOv2 probe |
| `d6ca8c7` | Tải checkpoint, pin revision, lockfile, script tái lập |
| `ef20700` | Demo và LP-FT |
| `a18dbf6`, `82ceb3a` | Phân tích lỗi v3, audit Lab split |
| `45755d5`, `7e75e42` | Tài liệu kỹ thuật, bảng rubric |
| `ab02c7e` | Ablation 24 epoch |
| `ccccef5` | Merge các bản sửa đợt 1 |
| `dfdf608` | Sửa bug theo hội đồng: demo, script tái lập, script so sánh, chặn ghi đè |
| `dd541b6`, `248a4b0` | Sửa tài liệu và viết theo giới hạn CI; CI cho metric phụ; sổ ghi đánh giá |
| `dbeaf09` → `991123a` | Đối chứng occlusion và baseline tâm ảnh |
| `0d47ea8`, `7c1a280`, `64d749b` | Khai báo trước E2 (Mixed 24 epoch), bản bổ sung seed 43–44 cùng nhánh Field-only, và bản sửa phép kiểm tái lập (commit trước mọi lần chạy) |
| `828bbfe` | Khai báo trước E3: AdaBN |
| `da2f057` | Hình, định nghĩa FO, audit Lab trong README |
| `034ad91` | Sửa câu chữ theo Leader; bảng occlusion trên cùng 378 ảnh; kết quả E2 seed 42 |
| `eede9dd` | Độ nhạy theo ranh giới nhãn BS/Septoria (B1) |
| `4a4c8ff`, `b1671cb` | Kết quả E3 AdaBN; lưu dự đoán từng ảnh để làm phép kiểm cặp |
| `6a6faca` → `53d3d20` | E2: 6 lần chạy 24 epoch, kiểm checkpoint niêm phong, tổng hợp 3 seed |
| `9e65ca4`, `e762937` | README: ranh giới BS/Septoria; RQ4 ở 24 epoch (câu chữ đã qua Leader duyệt) |
| `a04188a`, `02427c1` | REPO_REVIEW §h.2; sổ ghi đánh giá (104 lần đánh giá trên v3) |

Lý do của từng thay đổi được giải thích trong `REPO_REVIEW.md` §(h) và §(h.2) trên nhánh đợt 2.
