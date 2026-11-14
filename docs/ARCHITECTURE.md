# Kiến trúc Bộ Não AI Sinh Học (Bio-Inspired Multi-Region Brain)

## 1. Triết lý cốt lõi
Thay vì tiếp cận theo kỹ thuật Computer Vision truyền thống (`YOLO` / `DETR` / `ByteTrack` vẽ `bounding box` và gán `id=01`), hệ thống mô phỏng nguyên lý hoạt động âm thầm của não người:

1. **Không có `bounding box` hay `id=01`:** Đối tượng không phải một nhãn rời rạc. Sự tồn tại của "con mèo đang bay nhảy" là một **trạng thái hấp dẫn liên tục (Continuous Latent Trajectory)** gắn kết bởi chuyển động và đặc trưng thị giác.
2. **Xử lý âm thầm (Subconscious $\to$ Conscious Gating):** Hơn 90% hoạt động (lọc nền tĩnh, theo dõi chuyển động, trích xuất đặc trưng) diễn ra ngầm. Chỉ khi **Prediction Error (Độ bất ngờ)** vượt ngưỡng thích nghi, **Đồi thị (Thalamus)** mới phát tín hiệu đánh thức **Không gian làm việc toàn cục (Global Workspace)** và **Ký ức tình tiết**.
3. **Thị giác chủ động tiết kiệm năng lượng (Foveated Retina):** Tối ưu cho phần cứng `GTX 1650 Ti (4 GB VRAM)` + `23.84 GB RAM`. Ảnh thô được nén ~98% ngay tại võng mạc thành ngoại vi phân giải thấp (`96x96`) và điểm vàng sắc nét (`140x140`).

## 2. Sơ đồ phân vùng & Phần cứng

```
[Camera / Video / Sim]
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ KHỐI THỊ GIÁC (Visual Cortex - GPU GTX 1650 Ti FP32)     │
│                                                          │
│  1. DualResolutionRetina                                 │
│     ├─► Periphery (96x96)  ──► DorsalMotionStream        │
│     │                          (Conv-GRU Online Pred.)   │
│     │                               │                    │
│     │                        Surprise Field S(y,x)       │
│     │                               │                    │
│     │                          SuperiorColliculus        │
│     │                     (Saccade / Pursuit / IOR)      │
│     │                               │                    │
│     └─► Fovea (140x140) ◄── Điều khiển hướng nhìn (gx,gy)│
│             │  (Async 1-slot Mailbox + CUDA Stream)      │
│             ▼                                            │
│     VentralSemanticStream                                │
│     (DINOv2-S / Light Encoder + Temporal Binder)         │
└──────────────────────────┬───────────────────────────────┘
                           │ Async SensoryPackets
                           ▼
┌──────────────────────────────────────────────────────────┐
│ ĐỒI THỊ & KÝ ỨC (AsyncThalamusBus + Memory - CPU 8T/RAM) │
│  - Event-driven Global Latent Workspace & Familiarity    │
│  - EpisodicAttractorMemory (.npz Lưu/Quên sinh học)      │
│  - Hồi tiếp Top-Down Boredom xuống Colliculus & Thị giác │
│  - Giao tiếp chuẩn BrainRegion cho Âm thanh / Nội cảm    │
└──────────────────────────────────────────────────────────┘
```

## 3. Lộ trình 5 Chương / 10 Chặng Tiến Hóa Nhận Thức

### Chương 1 — Tri giác, Bất đồng bộ & Ký ức Sống
- **Chặng 1.1 (Xong):** Khối Thị giác vô thức (`brain/regions/vision/`) + Đồi thị (`brain/core/workspace.py`) + Ký ức liên kết (`brain/core/memory.py`).
- **Chặng 1.2 / 1.5 (Xong):** Kiến trúc Bất đồng bộ Đa nhịp (`AsyncVisualCortex` trên 2 CUDA Streams + `AsyncThalamusBus`).
- **Chặng 1.3 (Xong):**
  - **Ký ức có Lưu, Quên & Cố định dài hạn (*LTM Consolidation*):** Củng cố hố hút khi gặp lại, suy giảm và đào thải ký ức thoáng qua (`forgotten_count`), đồng thời **cố định bền vững** ký ức cốt lõi lặp lại (`hits >= 4` hoặc bền `>= 0.95`) xuống ngưỡng sàn dài hạn (`ltm_floor = 0.35`), kèm lưu/nạp liên phiên qua `.npz` (mặc định `data/episodic_memory.npz`).
  - **Chán quen Top-Down (*Familiarity Boredom*):** Hồi tiếp mức độ quen thuộc từ Đồi thị xuống `Colliculus` để giảm dính điểm nhìn trung tâm (*local lock*), tăng IOR tại chỗ và kích thích mắt khám phá ngoại vi mới mẻ.
- **Chặng 1.4 (Xong):**
  - **Ký ức ngữ nghĩa sạch & Đồ thị chuyển tiếp (`A → B`):** Loại bỏ nhiễu vận tốc khỏi vector ngữ nghĩa, áp dụng *Gaussian Foveal Pooling* (loại register tokens) cho `DINOv2`, lưu liên kết chuyển tiếp thưa có **suy giảm cạnh yếu (`transition_decay`)**, **gộp hố hút trôi trùng (`attractor merge`)**, dự đoán hố kế tiếp và phát hiện **Bất thường kỳ vọng (*Expectation Violation*)** khi cảnh quen đột ngột biến động.
  - **Tối ưu GTX 1650 Ti, Treo máy dài & Ghi nhớ xem lâu:** Lazy `fovea` GPU upload & `collect_diagnostics=False` ở headless, cổng **Foveal Deduplication** bỏ qua lần chạy `Ventral` nếu điểm vàng không đổi, shortcut khung hình tĩnh & mask zero-allocation ở `Dorsal`, lazy `FAISS` (`count >= 512`), đồng hồ tuổi ký ức `_decay_step`, **Extended STM Retention** (`grace_steps=450`, `decay_rate=0.00006`, `transition_decay=0.0006`) + **Silent Fixation Encoding** (ghi dấu nhẹ cảnh tĩnh lặng khi nhìn đủ lâu không cần giật mình), cùng **atomic `.npz` autosave (`--autosave-every`)** và thoát sạch khi đóng cửa sổ `[X]`.

### Chương 2 — Tưởng tượng & Giấc mơ (*Internal World Model*)
- **Chặng 2.1 — Tự sinh hình ảnh & Suy giảm Độ chính xác (*Visual Imagery & Precision Degradation*):** Giải mã ngược từ không gian ẩn (`Latent -> Visual Canvas`) giúp não tự "hình dung" vật thể trong trí nhớ khi không có kích thích từ camera, đồng thời mô hình hóa quá trình phai dần độ sắc nét (`_clarity`) / trôi về đại ý ước lượng (*Gist Drift & Reconsolidation*) thay vì xóa nhị phân 0/1.
- **Chặng 2.2 — Tự sinh chuyển động / Mơ (*Dreaming & Temporal Rollout*):** Mô phỏng quỹ đạo chuyển động theo thời gian trong trạng thái nhàn rỗi/ngủ để tái hiện và củng cố ký ức.

### Chương 3 — Ý thức Bản thân & Tự giám sát (*Self & Meta-Cognition*)
- **Chặng 3.1 — Hiểu bản thân là gì (*Minimal Self & Efference Copy*):** Phân biệt chuyển động do tự thân gây ra (*Self-caused*, như liếc mắt) với chuyển động của thế giới bên ngoài (*World-caused*).
- **Chặng 3.2 — Hiểu mình đang làm gì (*Meta-Cognition*):** Tự theo dõi trạng thái chú ý, mức độ bối rối/tự tin và nguyên nhân chuyển hướng quan tâm của chính hệ thống.

### Chương 4 — Nội cảm & Cảm xúc Cân bằng Nội môi (*Homeostasis & Affect*)
- **Chặng 4.1 — Bình thường (*Homeostasis Baseline*):** Trạng thái cân bằng năng lượng — bất ngờ — ký ức ở ngưỡng ổn định.
- **Chặng 4.2 — Vui vẻ / Hứng thú (*Curiosity & Reward*):** Tín hiệu phần thưởng nội tại khi khám phá được cấu trúc mới mà mô hình nội bộ học dần nắm bắt được.
- **Chặng 4.3 — Buồn bã / Thiếu kích thích (*Deprivation & Low Drive*):** Suy giảm động lực khi môi trường nghèo nàn thông tin hoặc cạn kiệt sự mới lạ dài hạn.
- **Chặng 4.4 — Đau đớn / Báo động sinh tồn (*Nociception & Conflict Signal*):** Xung báo động ưu tiên cao khi gặp xung đột cảm giác nặng, lỗi dự đoán mất kiểm soát hoặc nguy cơ vượt giới hạn an toàn.

### Chương 5 — Đa giác quan & Ngôn ngữ Nền tảng
- **Chặng 5.1 — Thính giác (`AuditoryCortex`):** Liên kết đa giác quan (hình ảnh + âm thanh) đồng bộ tại Đồi thị.
- **Chặng 5.2 — Ngôn ngữ nền tảng (*Grounded Language*):** Gắn nhãn ngôn ngữ từ trải nghiệm giác quan và trạng thái nội cảm (không tách rời thế giới thực).
