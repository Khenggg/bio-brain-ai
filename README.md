# Bio-Brain AI — Bộ Não Thị Giác Sinh Học (v1)

Hệ thống nghiên cứu bộ não AI đa khối hoạt động **âm thầm (subconscious-first)** theo nguyên lý sinh học: **không dùng bounding box** và **không gán object ID (`id=01`)**.

## Nguyên lý hoạt động
- **Võng mạc kép (`brain/regions/vision/retina.py`):** Nén khung hình thành ngoại vi độ phân giải thấp (`96x96`) và điểm vàng *fovea* sắc nét (`140x140`), tiết kiệm ~98.6% lượng điểm ảnh trên đầu vào Full HD.
- **Luồng Lưng dự đoán & triệt tiêu nền (`brain/regions/vision/dorsal.py`):** Dùng `Conv-GRU` học trực tuyến kết hợp bù trừ lia camera (2D FFT Phase Correlation) và thói quen hóa cục bộ để tự bỏ qua nền tĩnh hoặc chuyển động lặp lại, chỉ giữ lại bản đồ bất ngờ (*Surprise Field*).
- **Phản xạ mắt vô thức & Khám phá mới (`brain/regions/vision/colliculus.py`):** Điều khiển điểm nhìn *fovea* (*Saccade*, *Smooth Pursuit*, *Inhibition of Return*) từ trường năng lượng liên tục, kết hợp tín hiệu chán quen (*Top-Down Familiarity Boredom*) để mắt tự rời thứ đã quá quen sang khám phá vùng ngoại vi mới — không vẽ `bbox` hay đánh `id=01`.
- **Luồng Bụng & Gắn kết thời gian (`brain/regions/vision/ventral.py`):** Mã hóa vùng *fovea* bằng `facebook/dinov2-with-registers-small` (tự chọn `FP32` trên GPU `GTX` không có Tensor Core, `FP16 autocast` trên `RTX`, có fallback `light-conv`) kết hợp trạng thái liên tục theo thời gian.
- **Chế độ Bất đồng bộ Đa nhịp (`brain/regions/vision/async_cortex.py`, `brain/core/async_bus.py`):** `AsyncVisualCortex` tách vòng phản xạ mắt nhanh (`Retina + Dorsal + Colliculus`) chạy song song trên CUDA Stream riêng với luồng `Ventral` kích hoạt theo sự kiện (*saccade* / bất ngờ cao), đẩy xung bất đồng bộ về `AsyncThalamusBus`.
- **Đồi thị & Ký ức sống (`brain/core/`):** `ThalamusWorkspace` giữ >90% xử lý ở tầng vô thức, chỉ bừng sáng ý thức khi có biến cố bất ngờ/mới lạ, và phát hồi tiếp độ chán quen (*Familiarity Boredom*) xuống `Colliculus`. `EpisodicAttractorMemory` hỗ trợ củng cố khi gặp lại, quên sinh học (`forgotten_count`) cho hố ký ức bị bỏ quên, và lưu/nạp liên phiên ra file `.npz` (`--memory-path`, `--fresh-memory`, `--no-memory-io`).

## Cấu trúc thư mục
- `brain/core/`: Đồi thị (`workspace.py`), Event Bus bất đồng bộ (`async_bus.py`), Ký ức tình tiết (`memory.py`), và cấu trúc tín hiệu (`types.py`).
- `brain/regions/vision/`: Toàn bộ tiểu khối của Khối Thị Giác (`retina.py`, `dorsal.py`, `colliculus.py`, `ventral.py`, `cortex.py`, `async_cortex.py`).
- `brain/observer/`: Giao diện quan sát 4 khung hình trực tiếp (`visualizer.py`).
- `sim/cat_motion_env.py`: Phòng mô phỏng 2D có vật chuyển động bay nhảy và cây đung đưa (không cần dataset để test).
- `scripts/`: Bộ tải chống rớt mạng (`build_manifest.py`, `resilient_fetch.py`) và cài đặt offline (`install_offline.ps1`).
- `docs/ARCHITECTURE.md`: Kiến trúc tổng thể và lộ trình mở rộng đa khối.

## Cài đặt & Chạy thử (Windows / CUDA)
1. Tạo danh sách tải và tải toàn bộ thư viện + model `dinov2-with-registers-small` (tự nối tiếp khi rớt mạng/đổi IP):
   ```powershell
   python scripts\build_manifest.py
   python scripts\resilient_fetch.py
   ```
2. Cài đặt môi trường `.venv` 100% offline từ thư mục `wheels/`:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\install_offline.ps1
   ```
3. Chạy thử nghiệm trên mô phỏng, webcam hoặc video thật (thêm `--async-brain` để bật chế độ bất đồng bộ đa nhịp):
   ```powershell
   .\.venv\Scripts\python.exe run_vision_brain.py --source sim --async-brain
   .\.venv\Scripts\python.exe run_vision_brain.py --source webcam --async-brain
   .\.venv\Scripts\python.exe run_vision_brain.py --source path\to\video.mp4 --async-brain
   ```
4. Chạy bộ kiểm thử:
   ```powershell
   .\.venv\Scripts\pytest.exe -v
   ```
