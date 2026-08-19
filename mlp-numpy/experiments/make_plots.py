"""Vẽ toàn bộ biểu đồ từ logs/runs.csv + results/histories/*.json.

Đề yêu cầu TỐI THIỂU 6 biểu đồ. Danh sách gợi ý (đủ và bám sát yêu cầu):

  fig1  Loss/accuracy theo epoch — 4 hàm kích hoạt (cùng init, cùng preprocess)
  fig2  Phân bố activation theo tầng — lưới 4 (hàm kích hoạt) x L (tầng)
  fig3  Phân bố gradient theo tầng — cùng lưới, minh họa vanishing gradient
  fig4  Loss theo epoch — 5 cách khởi tạo (đường 'zeros' sẽ nằm phẳng)
  fig5  Tỉ lệ neuron chết (ReLU) / tỉ lệ bão hòa (sigmoid, tanh) theo epoch
  fig6  Loss theo epoch — 3 cách tiền xử lý
  fig7  Bar chart accuracy có error bar (mean ± std trên các seed)
  fig8  (BN) độ nhạy learning rate: val_acc theo lr, có BN vs không BN

Quy tắc: MỌI hình phải có tiêu đề, nhãn trục, đơn vị, chú giải — và phải
được nhắc tới trong nội dung báo cáo, nếu không sẽ mất điểm mục Báo cáo.
"""
# TODO: mỗi hình một hàm plot_figN(df, histories) -> lưu vào results/figures/figN_*.png
raise NotImplementedError
