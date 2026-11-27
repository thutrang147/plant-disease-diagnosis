# REPO_REVIEW — Đánh giá toàn diện repository `plant-disease-diagnosis`

Ngày review: 2026-11-27 · Nhánh được review: `main` @ `da5e1ba` · Phạm vi: read-only (không train, không inference, không sửa code/notebook/data).

Quy ước: **[VERIFIED]** = đã kiểm tra trực tiếp trong file (có đường dẫn / cell / số liệu tự tính lại). **[INFERENCE]** = suy luận / ý kiến chuyên môn, cần chủ repo xác nhận.

---

## (a) Executive summary

1. **Verdict tổng thể: mức "PASS WITH MAJOR REVISIONS" về mặt khoa học.** Kỹ thuật kiểm soát provenance (hash, seal, audit) tốt hơn hẳn mức thường gặp ở đồ án đại học. Nhưng **câu chuyện chính trong README đang nói quá so với bằng chứng ở hai chỗ quan trọng**.
2. **Mức tăng External v1 (30.65% → 53.23%, McNemar p=0.024) gần như hoàn toàn đến từ 16 ảnh Healthy tự chụp.** Trên 46 ảnh bệnh, ResNet Lab → Mixed chỉ đi từ 19/46 lên 21/46, còn ViT Lab → Mixed *giảm* từ 22/46 xuống 19/46 (Notebook 12) [VERIFIED]. Ngoài ra, con số "Early Blight 11/13" của baseline là kết quả của một model **gán nhãn Early Blight cho ~43/62 ảnh** [VERIFIED, file upstream raw-RGB].
3. **Không thể kết luận "Field-only > Mixed" ở RQ4.** Trên Field Test lệch nhau ròng chỉ 2 ảnh, bootstrap 95% CI của Δmacro-F1 là [−0.086, +0.023]. Trên Field Dev, tại đúng epoch đã chọn, Mixed ≥ Field-only ở cả 3 seed. Lab-retention, chiều đo mà Mixed thực sự có lợi trong lịch sử, lại không được đo [VERIFIED].
4. **Có nhiều confound chưa được nêu đủ.** Ví dụ ResNet fine-tune `layer4` (~75% tham số) trong khi ViT chỉ fine-tune block cuối (~8%). Tất cả các run đều chọn checkpoint ở **epoch cuối cùng** trong khi Dev vẫn đang tăng. Một ảnh trong Field Test (row 2524) là bản trùng pHash=0 với một ảnh train mang nhãn khác [VERIFIED].
5. **Cách sửa phần lớn là S/M effort, không cần train lại.** Gồm: viết lại khung kết luận, báo cáo theo strata, thêm CI/Holm, chạy inference Lab Val/Test cho 6 checkpoint NB15, commit các prediction CSV canonical, thêm LICENSE và dọn docs. Chỉ việc thu thập một locked field test mới là L.

---

## (b) Top 10 vấn đề ưu tiên

### #1 — External v1: mức tăng do strata Healthy bị confound với nguồn ảnh, không phải do phân biệt bệnh tốt hơn
- **Flaw:** README §2 và `docs/research_results_summary.md` đặt 53.23% vs 30.65% accuracy và p=0.0243 làm bằng chứng chính cho RQ2. Notebook 12 lại cho thấy toàn bộ mức tăng nằm ở 16 ảnh Healthy tự chụp. Trên 46 ảnh bệnh online: CNN Lab 19/46, CNN Mixed 21/46, ViT Lab 22/46, ViT Mixed 19/46.
- **Cơ chế:** Healthy ↔ self-collected và disease ↔ online là confound hoàn toàn. Cách giải thích hợp lý nhất là "model adapted học được rằng lá xanh chụp ngoài trời → Healthy". Đó không phải là nhận dạng bệnh tốt hơn. Trong bảng McNemar 24 vs 10, có 12 discordant đến từ Healthy (0→12/16).
- **Fix:** (1) Báo cáo External v1 **tách 2 strata** làm kết quả chính: Healthy recall (n=16) và 4-class disease accuracy / macro-F1 trên 46 ảnh. (2) Chạy McNemar riêng trên 46 ảnh bệnh. Dự đoán của tôi là p≈1, cần tính lại. (3) Đổi câu trong README thành: "Adaptation chủ yếu sửa lỗi baseline không bao giờ dự đoán Healthy; trên ảnh bệnh online không có cải thiện rõ ràng."
- **Effort:** S · **Impact:** High
- **[VERIFIED]** `notebooks/12_external_error_analysis.ipynb` (bảng `online_sources 46 · 19/46 22/46 21/46 19/46`; `self_collected 16 · 0/16 5/16 12/16 13/16`), `outputs/external_confirmatory_eval/confirmatory_paired_mcnemar_tests.csv`.

### #2 — Trade-off "Early Blight 11/13 → 1/13" phần lớn là artefact của một baseline thiên lệch
- **Flaw:** File prediction được track (loader upstream raw-RGB, cùng accuracy 19/62 với run canonical) cho thấy ResNet Lab dự đoán Early Blight cho 43/62 ảnh: 10 Healthy, 11 EB, 6 LB, 5 BS, 11 Septoria. Precision Early Blight ≈ 11/43 ≈ 0.26. Trên Field Dev, baseline này cũng có Early Blight recall cao bất thường (13/16 = 0.8125) trong khi Healthy và BS bằng 0 (`outputs/final_comparison_development/final_comparison_per_class.csv`). Ngược lại, ResNet Mixed gần như không bao giờ dự đoán EB (2/62).
- **Cơ chế:** Recall cao của một class "mặc định" không phải là năng lực phân biệt. Vì vậy câu "adaptation làm mất khả năng nhận Early Blight" bị phóng đại: thứ mất đi là một bias, cộng với một collapse ngược chiều ở Mixed.
- **Fix:** Đặt **prediction-distribution (cột tổng của confusion matrix) và per-class precision** bên cạnh recall trong README, NB11 và NB12. Viết lại thành "baseline sụp về Early Blight; Mixed gần như không dự đoán Early Blight. Cả hai đều thất bại với EB theo hai cách khác nhau."
- **Effort:** S · **Impact:** High
- **[VERIFIED]** `outputs/external_confirmatory_eval/predictions_model1_resnet18_lab_base_external_v1.csv` (crosstab tự tính). Run canonical EXIF nằm ngoài git nên tôi không kiểm được trực tiếp. `docs/experiment_reconciliation.md` ghi chỉ 4 nhãn ResNet Lab khác nhau (đều 2↔1), nên phân bố gần như giống [INFERENCE].

### #3 — RQ4: kết luận "Field-only cao hơn" không được dữ liệu hỗ trợ, và chiều đo quan trọng nhất bị bỏ
- **Flaw:**
  - (a) Paired discordance từng seed trên Field Test (Field-only-only / Mixed-only correct): seed 42 = 3/2, seed 43 = 1/0, seed 44 = 1/1. Chênh lệch ròng là 2 ảnh trên 141 cặp outcome.
  - (b) Tôi tự tính paired bootstrap trên 47 ảnh, lấy trung bình 3 seed: Δmacro-F1 (Mixed−FO) = −0.021, **95% CI [−0.086, +0.023]**; Δacc 95% CI [−0.071, +0.028].
  - (c) **Field Dev tại epoch đã chọn** (log NB15 archived): seed 42 FO 0.4388 / Mx 0.4390; seed 43 (ep7) 0.4361 / 0.4478; seed 44 0.4147 / 0.4327. Mixed ≥ FO ở cả 3 seed, tức là ngược chiều với Test.
  - (d) Lab retention không được đo. Trong lịch sử, đây lại là lợi ích rõ nhất của Mixed (A1 Lab Val 0.8185 vs R1 0.9556).
- **Cơ chế:** Hai tập đánh giá nhỏ cho hai chiều ngược nhau, nên kết luận trung thực là "không phát hiện khác biệt". Bỏ qua retention khiến RQ4 kiểm định Mixed đúng trên chiều mà nó không được kỳ vọng thắng.
- **Fix:** (1) Đổi kết luận README §5 thành: "Dưới protocol matched, không có bằng chứng Mixed cải thiện field performance (Δ trong CI chứa 0, Dev và Test ngược chiều)." (2) **Chạy inference Lab Val + Lab Test cho 6 checkpoint NB15**. Checkpoint đã import local theo `docs/notebook15_colab_guide.md`, đây chỉ là evaluation, không train, khoảng 1–2 giờ CPU. (3) Thêm bảng Dev @ selected epoch cạnh bảng Test. (4) Báo cáo paired bootstrap CI.
- **Effort:** S–M · **Impact:** High
- **[VERIFIED]** `data/results/notebook15/prediction_exports/*.csv` (tôi tự tính lại; macro-F1 khớp 6/6 giá trị gốc), `notebooks/archive/15_controlled_field_mixed_colab_executed.ipynb` (log "Seed 42 · epoch 8/8 · Field Dev F1 …"), `notebooks/08_mixed_domain_finetuning.ipynb` cell 18.

### #4 — "Historical Mixed advantage" mà NB15 bác bỏ là một strawman
- **Flaw:** Trong chính NB08, so sánh cùng init (R1 Mixed vs A1 Field-only, đều từ ResNet Lab baseline) **đã cho Mixed thấp hơn trên Field Dev** (0.3523 vs 0.4119). "Ưu thế" Mixed chỉ xuất hiện khi so Geometric+Mixed (R2 0.4459) với baseline. Nghĩa là ưu thế đó gồm geometric init cộng với việc chọn config tốt nhất trong 6 run (A1, A2, V1, R1, R2, V1-mixed) trên cùng 92 ảnh Dev.
- **Cơ chế:** Winner's curse. Khi chọn max trong K config trên một Dev nhỏ, điểm của config thắng bị thổi phồng. NB15 thực chất **xác nhận lại** kết quả R1 vs A1, chứ không lật ngược điều gì.
- **Fix:** Viết lại narrative RQ2–RQ4: "Lợi ích lịch sử của Mixed là retention (Lab Val 0.82 → 0.96), không phải field performance. NB15 cho thấy dưới matched exposure không có lợi ích field." Đây là câu chuyện mạch lạc và mạnh hơn bản hiện tại.
- **Effort:** S · **Impact:** High
- **[VERIFIED]** `notebooks/08_mixed_domain_finetuning.ipynb` cells 16–20, `notebooks/06_field_adaptation_verified.ipynb` cell 21.

### #5 — Selective reporting trong README: bỏ cột Field Test, nhấn kết quả tốt nhất trong nhiều so sánh, không hiệu chỉnh multiple comparisons
- **Flaw:**
  - Bảng "Main Results §1" chỉ có Lab Test / Field Dev / External v1 và in đậm ResNet Geo+Mixed. Cột Field Test có trong `docs/research_results_summary.md` thì cho ResNet Geo+Mixed 0.3587, **thấp hơn cả ViT Lab baseline 0.3617**, và McNemar ResNet Mixed vs Lab trên Field Test có p=0.79.
  - §4 chỉ trình bày ViT Mixed five-crop 38/62. Đây là ô tốt nhất trong 3 policy × 4 model × 2 dataset = 24 so sánh, và được chọn *sau khi* đã xem outcome. Ô ResNet Mixed / Field Dev / center 0.50 lại cho 0.4459 → 0.3603.
  - Có 8 McNemar trong `confirmatory_paired_mcnemar_tests.csv` nhưng chỉ một cái được nêu (p=0.024). Với Holm/Bonferroni (α/8 = 0.00625) cái này không còn significant.
- **Cơ chế:** Examiner sẽ thấy cherry-picking trước khi thấy các caveat rất cẩn thận nằm ở docs.
- **Fix:** (1) README §1 thêm cột Field Test (n=47) và in số đúng/tổng. (2) §4 trình bày cả bảng 4 model × 4 policy hoặc một heatmap, ghi rõ "exploratory, post-hoc". (3) Báo cáo Holm-adjusted p cho toàn bộ family test. (4) Bỏ in đậm "winner".
- **Effort:** S · **Impact:** High
- **[VERIFIED]** `README.md` §1/§4, `docs/research_results_summary.md` bảng RQ1–RQ2, `outputs/external_confirmatory_eval/confirmatory_paired_mcnemar_tests.csv`.

### #6 — Không có locked test độc lập; Field Test 47 ảnh đã bị xem trước và có nhãn nghi nhiễu
- **Flaw:** (1) Field Test là tập duy nhất của PlantDoc. Nó được inspect từ NB04, rồi tái dùng cho NB15 và cho reconciliation. (2) `data/splits/field_near_duplicate_candidates.csv`: train row **1760 (Septoria) ↔ test row 2524 (Bacterial Spot), pHash distance 0, same_class=False**. Train row đã bị quarantine, nhưng test row vẫn mang nhãn Bacterial Spot, và **cả 6 run NB15 đều dự đoán 2524 là Septoria**. Các cặp label conflict khác (Early↔Septoria, Septoria↔Bacterial) cho thấy PlantDoc có nhiễu nhãn hệ thống giữa các bệnh đốm lá. (3) Mỗi class trong test chỉ có 8–11 ảnh. Một ảnh Healthy đổi kết quả là recall đổi 12.5 pp.
- **Cơ chế:** Khi nhãn nhiễu rơi đúng vào cặp class mà model hay nhầm (BS→Septoria 8/9), "lỗi model" và "lỗi nhãn" không tách được.
- **Fix:** (S) Audit nhãn 47 ảnh test: 2 người gán nhãn độc lập, báo cáo Cohen's κ, và chạy lại metric với/không có các ảnh tranh chấp (không cần train). (L) Thu một **locked field test mới**: ≥30 ảnh/class, cùng nguồn chụp cho cả 5 class, group theo cây/vườn, pre-register protocol và chỉ mở một lần.
- **Effort:** S (audit) / L (test mới) · **Impact:** High
- **[VERIFIED]** CSV trên; `prediction_exports/*` (2524 → nhãn 4 ở 6/6 run). Câu "nhãn test 2524 sai" là **[INFERENCE]**.

### #7 — So sánh ResNet18 vs ViT-Small bị confound nhiều hơn mức "khác pretraining/transform" được nêu
- **Flaw:** Lúc adaptation (NB06–08, NB15), ResNet update `layer4 + fc`, 8,393,728 (+2,565 head) / 11,179,077 tham số, tức ~75% [VERIFIED: tôi đếm bằng torchvision trong container]. ViT chỉ update block cuối + norm + head = 1,777,157 / 21,667,589, tức ~8% [VERIFIED NB07 cell 14]. Ngoài ra batch 32 vs 16, ViT dùng AMP còn ResNet dùng fp32, lr baseline 1e-4 vs 1e-5, và AugReg in1k vs torchvision V1.
- **Cơ chế:** "ViT giữ Lab tốt hơn (0.9892 vs 0.8185)" có thể giải thích hoàn toàn bằng tỉ lệ tham số được update, không cần đến kiến trúc.
- **Fix:** Nêu rõ trong Limitations. Nếu có thời gian, chạy một control rẻ: ResNet chỉ update `layer4.1` + fc (~4.7M, gần ~40%), hoặc ViT update 6 block cuối, để tỉ lệ trainable gần nhau. Tối thiểu thêm một bảng "trainable params / tổng" cho mỗi run.
- **Effort:** S (doc) / M (control) · **Impact:** Med–High

### #8 — Mọi run đều dừng ở biên của budget: model under-trained, checkpoint selection không có ý nghĩa "early stopping"
- **Flaw:** NB06 A1/A2: selected epoch 6/6. NB07 V1: 5/6 (bằng 6/6). NB08 R1/R2/V1: 6/6. NB15: 8, 7, 8 trên 8. Field Dev vẫn đang tăng ở epoch cuối (ví dụ NB15 seed 44: 0.389 → 0.411 → 0.415 / 0.409 → 0.418 → 0.433). lr backbone 1e-5 với chỉ 192 updates.
- **Cơ chế:** Các so sánh (Field-only vs Mixed, ResNet vs ViT) được thực hiện ở **pha đầu của training**. Lab loss trong Mixed cạnh tranh gradient trong một budget nhỏ, nên kết quả có thể đổi khi train tới hội tụ.
- **Fix:** Ghi rõ là limitation. Thêm một ablation budget (ví dụ 16–24 epoch, 1 seed) để chứng minh plateau. Hoặc diễn giải RQ4 là "trong regime low-budget fine-tuning".
- **Effort:** S (doc) / M (ablation) · **Impact:** Med
- **[VERIFIED]** log các notebook nêu trên.

### #9 — Fresh clone không kiểm chứng được các con số headline; có số liệu mâu thuẫn trong các file được track
- **Flaw:**
  - Run canonical External v1 (`outputs/external_evaluation/...`), Healthy v2 (`outputs/healthy_field_evaluation/...`) và reconciliation (`outputs/research_integration/...`) **không nằm trong git**, nên các link trong `docs/research_results_summary.md` và `docs/experiment_reconciliation.md` bị gãy với người clone.
  - Thư mục được track `outputs/external_confirmatory_eval/` chứa số upstream raw-RGB **khác README**: ViT Lab 0.4516/0.4346 vs 0.4355/0.4157; ViT Mixed 0.5000/0.4710 vs 0.5161/0.4808; ResNet Lab macro-F1 0.2456 vs 0.2406; high-conf errors 26/43 vs 27/43. Examiner mở CSV sẽ thấy số khác README.
  - Ngoài ra: không có checkpoint; PlantVillage `requested_revision: "default"` chưa pin; `requirements.txt` không pin; còn đường dẫn tuyệt đối `/Volumes/Lexar E6/...` trong output NB04 và NB13.
- **Fix:** (1) Commit **prediction CSV canonical tối giản** (image_id, label, pred, probs, model sha; không có ảnh hay metadata cá nhân) cho External v1, Healthy v2 và Field Test 4-model vào `data/results/`. Lượng dữ liệu nhỏ, giống cách đã làm cho NB15. (2) Thêm `outputs/external_confirmatory_eval/SUPERSEDED.md` hoặc chuyển thư mục vào `outputs/archive/`. (3) Pin versions từ môi trường đã ghi (Python 3.12.13 / torch 2.10.0 cho training; 3.13.7 / torch 2.15.0 / timm 1.0.26 cho inference local) bằng `requirements-lock.txt`. (4) Phát hành 4 + 6 checkpoint trên GitHub Release / Zenodo / HF Hub, kèm SHA-256 đã có sẵn trong docs. (5) Pin PlantVillage commit SHA đang có trong HF cache.
- **Effort:** M · **Impact:** High
- **[VERIFIED]** `git ls-files outputs`, `.gitignore`, các CSV trên, `data/splits/dataset_sources.json`.

### #10 — Pháp lý và giấy phép: không có LICENSE; 62 ảnh (46 bên thứ ba) vẫn nằm trong git history
- **Flaw:** Repo không có `LICENSE` hay `CITATION.cff`. Commit `59e3bc4` thêm 62 ảnh External v1 (46 ảnh online có `redistribution_rights=not_established`). Ảnh đã bị gỡ khỏi tree nhưng **vẫn reachable trong history** (`git log` đếm được 63 file ảnh external được add). Các notebook 01, 05, 10 nhúng gallery PlantVillage/PlantDoc.
- **Cơ chế:** Nộp repo public thì đồng nghĩa phân phối lại ảnh bên thứ ba. Mặc định không có LICENSE nghĩa là "all rights reserved" cho code.
- **Fix:** (1) Thêm `LICENSE` cho code (MIT/Apache-2.0) và `DATA_NOTICE.md` ghi giấy phép từng dataset. PlantDoc là CC BY 4.0 theo repo gốc. Giấy phép PlantVillage cần kiểm chứng từ nguồn gốc; tôi chưa xác minh. (2) Với bản nộp: tạo **snapshot repo mới không có history** (hoặc `git filter-repo` có kế hoạch, sau khi đồng bộ với cộng tác viên), hoặc giữ repo private và chỉ chia sẻ với examiner. (3) Thêm `CITATION.cff` và trích dẫn đúng nguồn dataset/mirror (`geraldmc/*`).
- **Effort:** S–M · **Impact:** Med–High (rủi ro pháp lý khi public)
- **[VERIFIED]** không tìm thấy LICENSE; `git log --diff-filter=A`; `data/external_evaluation/tomato_field_eval_v2/README.md` §Attribution.

---

## (c) Chi tiết theo 9 chiều

### 1. Chất lượng câu hỏi nghiên cứu và độ mạch lạc RQ1–RQ4
- **RQ1 (domain gap):** rõ và được chứng minh mạnh. Lab macro-F1 0.97–0.99 so với Field 0.24–0.36 [VERIFIED NB04, NB09]. Nên bổ sung một cơ chế đã biết: PlantVillage có capture bias nặng. Noyan (2022) đạt 49% accuracy chỉ từ 8 pixel nền ([arXiv:2206.04374](https://arxiv.org/abs/2206.04374)). Thêm trích dẫn này giải thích *vì sao* gap lớn mà không cần thêm thí nghiệm.
- **RQ2:** đúng là có trade-off theo class, nhưng (#1, #2) cách đo hiện tại đang lẫn "học được Healthy" với "học được bệnh".
- **RQ3 (framing):** là RQ yếu nhất về logic. NB13 đo Healthy-only recall: một model luôn trả lời "Healthy" sẽ đạt 100%. NB14 đặt tên "Lesion resolution study" nhưng chính kết luận lại nói không đo resolution [VERIFIED NB14 cell 0 và 20]. Đề xuất: đổi tên thành "Inference-view sensitivity". Ghép Healthy recall (NB13) với **false-Healthy rate trên ảnh bệnh** để thành cặp sensitivity/specificity.
- **RQ4:** thiết kế control tốt (matched init/exposure/updates/seal). Kết luận cần sửa như #3/#4.
- **Through-line đề xuất [INFERENCE]:** "Lab accuracy không chuyển giao (RQ1). Supervised field data sửa được lỗi thô nhất (không bao giờ dự đoán Healthy) nhưng chưa sửa được phân biệt giữa các bệnh đốm lá (RQ2). Kết quả nhạy với framing (RQ3). Lab rehearsal giữ được Lab performance nhưng không tăng field performance khi đã control exposure (RQ4)." Câu chuyện này trung thực hơn và vẫn có đóng góp.
- **Under-claiming:** công phu về provenance và audit (seal hash trước khi mở test, hash tensor từng batch) là điểm mạnh hiếm có, nhưng README không nhấn mạnh. Nên đưa vào Contributions dưới dạng "evaluation hygiene".

### 2. Thiết kế thí nghiệm và validity
- **Splits:** Lab dùng StratifiedGroupKFold theo `leaf_id`. Tuy nhiên 1,510 (train) / 378 (val) / 465 (test) ảnh không có leaf group, mỗi ảnh được gán `unknown_row:<id>` riêng [VERIFIED `data/splits/lab_*.csv`]. Rủi ro cùng một lá nằm ở nhiều partition, nên Lab Test có thể bị thổi phồng [INFERENCE]. Fix S: chạy pHash grouping cho phần ungrouped, giống đã làm với PlantDoc.
- **Field split:** pHash ≤4 connected components, quarantine 14 dòng; nhất quán [VERIFIED `field_dev_protocol.json`]. Ngưỡng ≤4 là bảo thủ; nên báo cáo phân bố khoảng cách min (5–12) để chứng minh không còn near-dup.
- **Selection trên Field Dev:** 6+ config được so trên cùng 92 ảnh (#4). Chọn "best model" rồi báo điểm trên chính Dev đó là optimistic. Tài liệu có ghi nhận, nhưng README vẫn in đậm.
- **Test reuse:** xem #6.
- **Confounds External v1/v2:** source ≡ class (#1). Thêm vào đó, Healthy v2 có 49 crop từ 23 ảnh cha (21 scene), nên n hiệu dụng ≈ 21–23, không phải 57 [VERIFIED NB13 cell 19].
- **Seeds:** lịch sử là 1 seed. NB15 có 3 seed nhưng dùng chung test, nên SD chỉ phản ánh training noise [VERIFIED; tài liệu đã nêu đúng].
- **Power [INFERENCE, tính nhanh]:** với n=47, Wilson 95% CI cho 22/47 khoảng [0.33, 0.61], rộng ~±14 pp. Để phát hiện Δaccuracy 5 pp với paired McNemar (tỉ lệ discordant ~10%) cần cỡ vài trăm ảnh. **Không thiết kế nào trong repo có power để phát hiện hiệu ứng < 10 pp.** Nên nói thẳng điều này trong phần defense.
- **McNemar:** dùng exact; đúng. Thiếu hiệu chỉnh multiple comparisons (#5). NB04 p=0.3877 được báo cáo đúng là không significant.
- **Effect sizes:** nên báo Δ kèm CI (bootstrap theo ảnh; theo parent/scene cho Healthy v2), thay vì chỉ p.

### 3. Metrics và reporting
- **Macro-F1 trên mẫu số nhỏ:** Field Test có Healthy 8, BS 9. Một ảnh làm F1 của class đổi tới ~0.1, tức macro-F1 đổi ~0.02. Đây đúng bằng cỡ Δ của RQ4 (−0.0228). Nên báo **correct/total** cho mỗi class (NB15 đã làm, tốt) và thêm **balanced accuracy**, metric dễ giải thích hơn macro-F1 khi n nhỏ.
- **Healthy-only recall:** là sensitivity cho một class, không phải five-class performance. README §3 trình bày đúng là Healthy-only, nhưng nên thêm specificity tương ứng (#1).
- **Parent-balanced recall** (NB13): ý tưởng đúng. Nên kèm cluster bootstrap CI theo scene.
- **ECE** (`outputs/statistical_analysis/calibration_and_confidence_metrics.csv`, `confirmatory_summary_all_partitions.csv`): 10 bins trên 47–62 mẫu là rất không ổn định. Nên dùng reliability diagram + Brier score, hoặc bỏ.
- **Sai lệch số liệu phát hiện được:**

| # | Nơi A | Nơi B | Khác biệt | Đánh giá |
|---|---|---|---|---|
| 1 | README §1–2 (canonical EXIF) | `outputs/external_confirmatory_eval/confirmatory_summary_all_partitions.csv` (tracked) | ViT Lab acc 0.4355 vs 0.4516; ViT Mixed 0.5161 vs 0.5000; ResNet Lab F1 0.2406 vs 0.2456 | Có giải thích trong `experiment_reconciliation.md`, nhưng file tracked không ghi "superseded" [VERIFIED] |
| 2 | `research_results_summary.md`: high-conf errors 27/43 | CSV tracked: 26/43 (0.6047) | cùng nguyên nhân EXIF | [VERIFIED] |
| 3 | README §1 chỉ có 3 cột | `research_results_summary.md` có thêm Field Test | thiếu cột chứ không sai số; xem #5 | [VERIFIED] |
| 4 | `docs/cleanup.md` "ten-notebook record", "Four local checkpoints" | hiện có 15 notebook; `workflow.md` ghi 6 checkpoint | doc lịch sử chưa được tách khỏi doc hiện hành | [VERIFIED] |
| 5 | NB14 tên "lesion_resolution_study" | nội dung NB14 tự phủ nhận "resolution" | đặt tên gây hiểu nhầm | [VERIFIED] |
| 6 | `outputs/statistical_analysis/accuracy_confidence_intervals.csv` | — | bootstrap F1 CI để trống cho 6/14 dòng (dữ liệu hardcode làm tròn trong `src/statistical_analysis.py` dòng 122–144) | [VERIFIED] |

Tôi đã đối chiếu và thấy khớp: Lab Test, Field Dev (NB09 ↔ `final_comparison_summary.csv` ↔ README), Field Test NB15 (6 macro-F1 tự tính ↔ `original_field_test_summary.csv` ↔ README), Healthy 0/15/44/43 ↔ `submission_audit.md`, và split counts 5365/1341/1692/370/92/47/14.

### 4. Thí nghiệm / phân tích bổ sung có giá trị nhất (xếp theo impact/effort)

| Hạng | Việc | Effort | Impact | Ghi chú |
|---|---|---|---|---|
| 1 | Báo cáo External v1 theo strata + McNemar riêng cho 46 ảnh bệnh | S | High | chỉ cần các prediction CSV đã có |
| 2 | Paired bootstrap CI + Holm cho mọi so sánh chính | S | High | `src/statistical_analysis.py` đã có sẵn hàm |
| 3 | Lab Val/Test inference cho 6 checkpoint NB15 | S–M | High | trả lời RQ4 đầy đủ (retention) |
| 4 | Audit nhãn 47 ảnh Field Test (2 annotator, κ) + sensitivity analysis | S | High | đặc biệt các cặp BS↔Septoria |
| 5 | Bảng Dev @ selected epoch cạnh Test cho NB15 | S | Med | số đã có trong log |
| 6 | Specificity: tỉ lệ ảnh bệnh bị dự đoán Healthy, cạnh Healthy recall NB13 | S | Med | |
| 7 | pHash grouping cho Lab ungrouped + đánh giá lại Lab Test | S–M | Med | |
| 8 | Budget ablation (16–24 epoch, 1 seed) | M | Med | xử lý #8 |
| 9 | ViT controlled replication của NB15 (3 seed) | M | Med | khái quát RQ4 vượt ra ngoài ResNet; có thể dùng lại code `src/controlled_field_mixed.py` |
| 10 | Background-only / lesion-masked probe trên PlantDoc (giống Noyan) | M | Med | bằng chứng định lượng cho shortcut thay cho Grad-CAM định tính |
| 11 | Lesion annotation (bbox) cho ~50 ảnh Dev + pointing-game cho Grad-CAM | M | Med | biến NB10 từ định tính thành định lượng |
| 12 | Locked field test mới, cùng nguồn cho 5 class | L | Very High | cách duy nhất để đưa ra claim generalization |

### 5. Reproducibility và provenance
- **Điểm mạnh [VERIFIED]:** SHA-256 cho split, checkpoint, config; NB15 có evaluation seal trước khi mở test (`original_evaluation_seal.json`); audit read-only; notebook ghi git revision.
- **Fresh clone làm được:** đọc notebook đã lưu output; chạy lại NB04 và NB15 report từ CSV; chạy tests/audit.
- **Fresh clone không làm được:** mọi inference (không có checkpoint), NB10–14 (dữ liệu private, output ignored), training (cần Drive), External v1/v2 (manifest private).
- **Thiếu:** pin PlantVillage revision; lock môi trường; release checkpoint; prediction CSV canonical cho NB11–14; các link docs đang trỏ vào thư mục ignored (#9).
- Nhiều notebook "kéo code từ GitHub" bằng `git pull --ff-only` (NB02 cell 4). Nếu chạy lại sẽ dùng HEAD hiện tại, không phải revision đã ghi. Nên đổi sang checkout theo SHA đã ghi (NB15 archive đã có `GIT_REF`; áp dụng tương tự cho các notebook còn lại).

### 6. Code quality
- `src/` có hai lớp rõ rệt. Lớp research hiện hành (`dataset.py`, `evaluate_*`, `controlled_field_mixed.py`, `statistical_analysis.py`, `reconcile_research_results.py`) và lớp scaffold không dùng (`src/train.py` chỉ raise `NotImplementedError`; `src/models.py` là ViT-B/16 không phải ViT-Small; `configs/config.yaml` không được đọc; `app/streamlit_app.py` chỉ có upload) [VERIFIED]. Examiner sẽ hỏi "vì sao có train.py không chạy được?". **Fix S:** chuyển cả 4 file vào `archive/scaffold/` hoặc xoá, kèm 1 dòng trong README.
- **Logic training trùng lặp:** ít nhất 5 bản `run_epoch`/train loop trong notebook 02, 03, 05, 06, 07, 08 và `src/controlled_field_mixed.py`. **Fix M:** chỉ làm nếu viết tiếp; không nên refactor trước khi nộp vì sẽ phá provenance.
- `src/statistical_analysis.py` hardcode số đã làm tròn từ notebook (dòng 122–144). Nên đọc từ CSV khi có.
- **Tests:** 15 test (theo `docs/submission_audit.md`) phủ dataset loader, external audit, statistical export, NB15 report. Chưa có test cho metric core (macro-F1 với class vắng, McNemar, Wilson) và cho `controlled_field_mixed` selection rule (đã có 1 file). Đề xuất thêm property test nhỏ cho `mcnemar_exact`, `wilson_ci` và `compute_ece`.
- **Archive hygiene:** `notebooks/archive/06_field_adaptation_grouped.ipynb` nặng 5.1 MB, `notebooks/10_error_analysis.ipynb` 7.8 MB, `01` 6.4 MB do ảnh nhúng. `.git` = 115 MB. Có thể chấp nhận, nhưng ảnh nhúng chính là vấn đề license (#10).
- **Kết quả chạy test:** xem mục (f).

### 7. Documentation
- **README:** cấu trúc tốt (Abstract, RQ, Data, Results, Limitations), nhưng có các vấn đề selective reporting (#1, #2, #5).
- **Doc sprawl:** có 7 file `docs/` (~124 KB). Trong đó 4 file là **log quá trình của agent/cleanup** (`cleanup.md`, `documentation_audit.md`, `submission_audit.md`, `experiment_reconciliation.md`), chứa lệnh git thủ công, tên nhánh, "Case B, with justified Case C corrections", "byte-for-byte", "protected files". Examiner không cần những nội dung này, và chúng làm loãng tín hiệu.
- **Fix S–M:**
  - Giữ `research_results_summary.md` (đổi tên `RESULTS.md`) và `workflow.md` (đổi tên `REPRODUCIBILITY.md`).
  - Viết mới `METHODS.md` khoảng 2 trang: splits, training recipes dạng bảng (lr, batch, trainable params, epochs, selection), evaluation.
  - Chuyển 4 file log vào `docs/archive/` kèm 1 câu giải thích.
  - `notebook15_colab_guide.md` giữ nguyên, nhưng phần "Ready-to-upload files" đưa vào archive.
- **Jargon nội bộ cần thay:** "historical", "canonical", "upstream", "verified" (dùng với nghĩa nội bộ); tên model "R2/V1/A1". Nên có bảng thuật ngữ, hoặc đổi sang tên mô tả như "ResNet18 · geometric init · mixed".

### 8. Data ethics và pháp lý
- Xem #10.
- Ảnh tự chụp: có metadata EXIF (GPS?) trong file gốc. Nếu từng public trong history thì cần kiểm tra EXIF GPS [INFERENCE; tôi không mở được ảnh trong clone này].
- Human review crop Healthy không ghi tên hay ngày reviewer [VERIFIED v2 README]. Nên ghi ít nhất vai trò reviewer và ngày.
- PlantDoc: CC BY 4.0 theo [repo gốc](https://github.com/pratikkayal/PlantDoc-Dataset), nên phải attribution. Mirror `geraldmc/plantdoc-full` cũng phải được trích dẫn.

### 9. Mức sẵn sàng nộp và bảo vệ
- **Rủi ro cao nhất khi bảo vệ:** #1 (Healthy confound), #3 (RQ4 CI chứa 0), #5 (cherry-picking), #6 (test reuse). Cả 4 đều sửa được bằng cách viết lại, cộng một vài phân tích S trước buổi bảo vệ.
- Xem phần (e) cho câu hỏi và câu trả lời đề xuất.

---

## (d) Kế hoạch hành động

### Làm trong tuần này (không cần train lại)
1. README: thêm cột Field Test; tách External v1 thành 2 strata; thêm prediction distribution cho Early Blight; bỏ in đậm winner (#1, #2, #5).
2. Viết lại kết luận RQ2/RQ4 theo through-line ở §c.1; nêu R1 vs A1 (#3, #4).
3. Tính: McNemar 46 ảnh bệnh; paired bootstrap CI cho NB15 (số tôi đã tính có thể dùng làm kiểm tra chéo); Holm cho các McNemar.
4. Inference Lab Val/Test cho 6 checkpoint NB15 (#3).
5. Đánh dấu `outputs/external_confirmatory_eval/` là superseded; commit prediction CSV canonical tối giản (#9).
6. Thêm `LICENSE`, `DATA_NOTICE.md`, `CITATION.cff`; quyết định phương án history/snapshot cho bản nộp (#10).
7. Chuyển scaffold và 4 file log docs vào archive; đổi tên NB14 hoặc tiêu đề của nó.
8. Bảng recipe (trainable params, lr, batch, AMP, epochs, selected epoch) trong `METHODS.md` (#7, #8).

### Làm nếu còn thời gian
- Audit nhãn 47 ảnh test + sensitivity (#6, S).
- pHash grouping cho Lab ungrouped.
- Budget ablation; ViT controlled replication; background-only probe; lesion bbox + pointing game.
- Locked field test mới (L): đây là việc duy nhất nâng được claim từ "descriptive" lên "generalization".

---

## (e) Câu hỏi examiner có thể hỏi và gợi ý trả lời trung thực

1. **"External v1 tăng 23 pp. Có phải chỉ vì model học được 'ảnh tự chụp = Healthy'?"**
   *Trả lời:* "Đúng là gần như toàn bộ mức tăng nằm ở 16 ảnh Healthy tự chụp. Trên 46 ảnh bệnh online, ResNet chỉ đi từ 19 lên 21, còn ViT giảm từ 22 xuống 19. Vì nguồn và class trùng nhau, chúng tôi không coi External v1 là benchmark five-class mà báo cáo tách strata."
2. **"Baseline nhận 11/13 Early Blight tốt hơn adapted model. Sao lại nói adaptation tốt?"**
   *Trả lời:* "Baseline dự đoán Early Blight cho khoảng 2/3 số ảnh, precision ~0.26. Đó là bias chứ không phải năng lực. Model adapted thì gần như không dự đoán Early Blight. Cả hai thất bại với Early Blight theo hai kiểu khác nhau, và chúng tôi báo cáo điều này như một failure mode mở."
3. **"Field-only hơn Mixed 0.023 macro-F1 trên 47 ảnh. Có significant không?"**
   *Trả lời:* "Không. Chênh lệch ròng là 2 ảnh, bootstrap CI [−0.086, +0.023], và trên Field Dev chiều còn ngược lại. Kết luận của chúng tôi là không có bằng chứng Mixed cải thiện field performance khi đã control exposure; chúng tôi không kết luận Field-only tốt hơn."
4. **"Test set 47 ảnh đã xem từ notebook 04. Có leakage vào quyết định không?"**
   *Trả lời:* "Có xem từ trước, nên chúng tôi gọi đó là historical test. Mọi selection dùng Field Dev, và NB15 seal hash checkpoint trước khi mở test. Claim generalization cần một locked test mới; đây là limitation chính."
5. **"ViT giữ Lab tốt hơn ResNet có phải nhờ kiến trúc?"**
   *Trả lời:* "Không thể kết luận. ResNet update ~75% tham số (layer4), ViT ~8% (block cuối); ngoài ra pretraining, batch, AMP cũng khác. Đây là so sánh giữa hai configuration hoàn chỉnh."
6. **"Tại sao checkpoint luôn ở epoch cuối? Model đã hội tụ chưa?"**
   *Trả lời:* "Chưa; Dev vẫn đang tăng. Kết quả thuộc regime low-budget fine-tuning (192 updates). Chúng tôi nêu đây là limitation và đề xuất budget ablation."
7. **"Grad-CAM chứng minh model nhìn vào vết bệnh?"**
   *Trả lời:* "Không. Không có lesion annotation, nên Grad-CAM và attention chỉ là minh hoạ định tính; chúng tôi không đưa ra claim định vị."
8. **"Five-crop tăng 32 → 38/62, sao không dùng làm policy?"**
   *Trả lời:* "Đó là ô tốt nhất trong 24 so sánh, được chọn sau khi đã thấy outcome, và có ô khác giảm (ResNet center-0.5 trên Dev). Chúng tôi chỉ kết luận rằng model nhạy với framing."
9. **"PlantVillage 99% có đáng tin không?"**
   *Trả lời:* "Chỉ đáng tin trong phạm vi PlantVillage. Dataset này có capture bias đã được ghi nhận (Noyan 2022: 49% accuracy chỉ từ 8 pixel nền), và khoảng 28% ảnh Lab không có leaf group nên có thể còn near-duplicate giữa các split."
10. **"Ảnh trong repo có quyền phân phối không?"**
    *Trả lời:* "Ảnh online có `redistribution_rights=not_established` và không còn trong tree hiện tại. Bản nộp là snapshot không có history, hoặc chia sẻ private. PlantDoc là CC BY 4.0 và được attribution."
11. **"n=57 Healthy v2 có phải 57 quan sát độc lập?"**
    *Trả lời:* "Không. 49 crop đến từ 23 ảnh cha trong 21 scene. Chúng tôi báo cáo parent-balanced recall, và n hiệu dụng khoảng 21–23."

---

## (f) File đã kiểm tra và những gì không kiểm tra được

**Đã đọc trực tiếp:**
- `README.md`; toàn bộ `docs/*.md` (7 file).
- `data/splits/*.json` và toàn bộ CSV: header, đếm dòng, overlap group, class counts, quarantine, near-duplicate.
- `data/results/notebook15/*`: README, summary, seal, config; 6 prediction export đã tính lại metric và bootstrap.
- `data/external_evaluation/tomato_field_eval_v2/README.md` và `class_mapping.json`. `build_dataset.py` và `validate_dataset.py` chỉ đọc lướt cấu trúc.
- `outputs/*` được track: summary, per-class, McNemar, CI, prediction CSV External v1 upstream.
- `src/train.py`, `src/models.py`, `configs/config.yaml`, `app/streamlit_app.py`; danh sách hàm và hardcode trong `src/statistical_analysis.py`. Các module `src/` khác chỉ kiểm qua kích thước và tham chiếu trong docs, không đọc từng dòng.
- `tests/*`: số lượng test.
- `.gitignore`, git history (commit thêm ảnh, kích thước `.git`).
- **15 notebook canonical:** parse JSON, đọc markdown và text output đã lưu. Code cell đọc kỹ ở 01, 02, 06 (optimizer), 08 (optimizer), còn lại đọc lướt.
- **Archive:** `15_controlled_field_mixed_colab_executed.ipynb` (log Dev F1 theo epoch). Ba notebook archive còn lại (06 old, 06 grouped, 11 combined) chỉ được parse, chưa review nội dung.

**Không kiểm tra được:**
- Checkpoint (`models/` không có trong clone).
- Các output bị ignore: run canonical External v1 `20261125_163347_4ca8183f`, Healthy v2, reconciliation, NB15 imported run.
- Manifest/ảnh private External v1/v2; Google Drive artifacts.
- Nội dung ảnh nhúng trong notebook (không xem bằng mắt).
- Giấy phép PlantVillage (chưa xác minh).

### Phụ lục — kết quả chạy tests/audit (fresh clone, cloud container, `HF_*_OFFLINE=1`)
- `python -m unittest discover -s tests -v`: **Ran 15 tests, OK** (0 failed) [VERIFIED].
- `python scripts/audit_repository.py`: **exit 0, `"status": "passed"`**. 30 Python file parse được, 21 module `src` import được, split sizes khớp. Tuy nhiên chỉ **17 prediction file** được kiểm, so với 98 trong `docs/submission_audit.md`. 81 file còn lại nằm trong output bị ignore, đây là minh hoạ cụ thể cho #9 [VERIFIED].
- Test pass chỉ chứng minh evidence đã lưu là nhất quán và I/O an toàn. Nó không chứng minh các kết luận khoa học ở #1–#8 là đúng.
