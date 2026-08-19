"""Phần 'Yêu cầu khác': đối chiếu bản NumPy với bản PyTorch tương đương
để kiểm chứng tính đúng đắn.

Cách đối chiếu thuyết phục nhất (mạnh hơn là chỉ so accuracy cuối cùng):
  1. Dựng nn.Sequential cùng kiến trúc.
  2. COPY trọng số từ mô hình NumPy sang PyTorch (nhớ transpose: NumPy dùng
     W (D_in, D_out), torch.nn.Linear dùng weight (D_out, D_in)).
  3. Cho cùng một batch đi qua cả hai -> so logits (sai khác ~1e-10).
  4. backward cả hai -> so gradient từng tầng.
  5. Chạy vài bước SGD với cùng lr -> so loss theo từng bước.
"""
raise NotImplementedError
