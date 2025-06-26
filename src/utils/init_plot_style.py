# import matplotlib as mpl
# import seaborn as sns
# This is NMI style.

def init_style(mpl, sns):
    sns.set_style("ticks")
    # 将毫米转换为英寸（1 mm = 0.0393701 英寸）
    mm_to_inch = 0.0393701

    # 设置全局参数
    mpl.rcParams.update({
        # 图像分辨率
        'figure.dpi': 300,  # 设置分辨率为 300 dpi
        'savefig.dpi': 300,  # 保存图像时使用 300 dpi

        # 图像尺寸（最大宽度 180 mm）
        'figure.figsize': (180 * mm_to_inch, 180 * mm_to_inch * 0.75),  # 宽 180 mm，高按比例设置（如 4:3 宽高比）

        # 字体设置
        'font.family': 'sans-serif',  # 使用无衬线字体
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],  # 优先使用 Arial 或 Helvetica
        'font.size': 6,  # 设置标准字体大小为 6 pt（在 5–7 pt 范围内）

        # 希腊字符使用 Symbol 字体
        'mathtext.fontset': 'custom',  # 允许自定义数学字体
        'mathtext.rm': 'Arial',  # 常规数学文本使用 Arial
        'mathtext.it': 'Arial:italic',  # 斜体使用 Arial 斜体
        'mathtext.bf': 'Arial:bold',  # 粗体使用 Arial 粗体
        'mathtext.sf': 'Symbol',  # 希腊字符等使用 Symbol 字体

        # 图像背景和边框
        'figure.facecolor': 'white',  # 白色背景
        'axes.facecolor': 'white',  # 轴区域白色背景
        'savefig.facecolor': 'white',  # 保存时白色背景
        'savefig.bbox': 'tight',  # 紧凑边界框，减少多余空白
        'savefig.pad_inches': 0.05,  # 边界填充

        # 线条和标记
        'lines.linewidth': 1.0,  # 线条宽度
        'lines.markersize': 4,  # 标记大小

        # 坐标轴刻度设置
        'xtick.direction': 'out',  # x 轴刻度向外
        'ytick.direction': 'out',  # y 轴刻度向外
        'xtick.labelsize': 7,  # x 轴刻度数字大小为 3 pt
        'ytick.labelsize': 7,  # y 轴刻度数字大小为 3 pt

        # 禁用网格
        'axes.grid': False,  # 关闭背景网格

        'axes.labelsize': 6,
        'figure.titlesize': 6,

        # 禁用legend外框
        'legend.frameon': False,

    })
# 最大宽度
# 180 * mm_to_inch
